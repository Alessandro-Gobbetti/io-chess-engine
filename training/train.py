"""
@file train.py
@brief Training script for model.py (factorized MoE).

3-Phase training pipeline:
  Phase 1: Base training with teacher routing weights
  Phase 2: Expert specialization on expert-specific datasets
  Phase 4: Joint fine-tuning with top-2 routing

This script is intentionally styled after train_light_moe.py while using
factorized packed inputs from dataset.py.
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import ChessExpertFactorizedDataset, ChessMoEFactorizedDataset
from loss import base_loss_components
from model import ChessNetFactorizedMoE
from utils import Colors, set_seed

C = Colors


class ChunkedRandomSampler(Sampler):
    """Shuffle sample order in large chunks to favor sequential disk I/O."""

    def __init__(self, data_source, chunk_size=1_000_000):
        self.num_samples = len(data_source)
        self.chunk_size = chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.num_chunks = math.ceil(self.num_samples / self.chunk_size)

    def __iter__(self):
        chunk_indices = torch.randperm(self.num_chunks).tolist()
        for chunk_idx in chunk_indices:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.num_samples)
            local_indices = torch.randperm(end_idx - start_idx).tolist()
            for idx in local_indices:
                yield start_idx + idx

    def __len__(self):
        return self.num_samples


class ExpertChunkedSampler(Sampler):
    """Spatially sort expert indices, then randomize at chunk/local levels."""

    def __init__(self, dataset_indices, chunk_size=1_000_000):
        self.num_samples = len(dataset_indices)
        self.chunk_size = chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.num_chunks = math.ceil(self.num_samples / self.chunk_size)

        print("Sorting expert indices to optimize disk I/O...")
        sorted_positions = np.argsort(dataset_indices)

        self.spatial_chunks = []
        for i in range(self.num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self.num_samples)
            self.spatial_chunks.append(sorted_positions[start:end])
        print("Done sorting.")

    def __iter__(self):
        chunk_order = torch.randperm(self.num_chunks).tolist()
        for c_idx in chunk_order:
            chunk = self.spatial_chunks[c_idx]
            local_order = torch.randperm(len(chunk)).tolist()
            for idx in local_order:
                yield int(chunk[idx])

    def __len__(self):
        return self.num_samples


class ConcatChunkedRandomSampler(Sampler):
    """Chunked shuffle that stays shard-local for ConcatDataset I/O locality."""

    def __init__(self, datasets, chunk_size=1_000_000):
        self.chunk_size = chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.shard_lengths = [len(ds) for ds in datasets]
        self.shard_offsets = []
        self.chunk_descriptors = []

        total = 0
        for shard_idx, shard_len in enumerate(self.shard_lengths):
            self.shard_offsets.append(total)
            total += shard_len

            num_chunks = math.ceil(shard_len / self.chunk_size)
            for c_idx in range(num_chunks):
                start = c_idx * self.chunk_size
                end = min(start + self.chunk_size, shard_len)
                self.chunk_descriptors.append((shard_idx, start, end))

        self.num_samples = total

    def __iter__(self):
        chunk_order = torch.randperm(len(self.chunk_descriptors)).tolist()
        for c_idx in chunk_order:
            shard_idx, start, end = self.chunk_descriptors[c_idx]
            local_order = torch.randperm(end - start).tolist()
            shard_offset = self.shard_offsets[shard_idx]
            base = shard_offset + start
            for idx in local_order:
                yield base + idx

    def __len__(self):
        return self.num_samples


class ConcatExpertChunkedSampler(Sampler):
    """Expert sampler for ConcatDataset preserving per-shard spatial locality."""

    def __init__(self, expert_datasets, chunk_size=1_000_000):
        self.chunk_size = chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.shard_offsets = []
        self.spatial_chunks = []

        total = 0
        print("Sorting expert indices to optimize disk I/O across shards...")
        for shard_idx, ds in enumerate(expert_datasets):
            if ds.mode != "indices":
                raise ValueError("ConcatExpertChunkedSampler requires index-mode expert datasets")

            self.shard_offsets.append(total)
            total += len(ds)

            print(f"  - shard {shard_idx + 1}/{len(expert_datasets)}: {len(ds):,} samples")
            sorted_positions = np.argsort(ds.indices)
            num_chunks = math.ceil(len(ds) / self.chunk_size)
            for c_idx in range(num_chunks):
                start = c_idx * self.chunk_size
                end = min(start + self.chunk_size, len(ds))
                self.spatial_chunks.append((shard_idx, sorted_positions[start:end]))
        print("Done sorting.")

        self.num_samples = total

    def __iter__(self):
        chunk_order = torch.randperm(len(self.spatial_chunks)).tolist()
        for c_idx in chunk_order:
            shard_idx, chunk = self.spatial_chunks[c_idx]
            local_order = torch.randperm(len(chunk)).tolist()
            shard_offset = self.shard_offsets[shard_idx]
            for idx in local_order:
                yield shard_offset + int(chunk[idx])

    def __len__(self):
        return self.num_samples


def print_header(text):
    print(f"\n{C.CYAN}{C.BOLD}{'=' * 70}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {text}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'=' * 70}{C.RESET}")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(args, phase_name):
    if not args.wandb and not args.wandb_offline:
        return None

    if not WANDB_AVAILABLE:
        print(f"{C.YELLOW}[WARN] wandb not installed, skipping logging{C.RESET}")
        return None

    mode = "offline" if args.wandb_offline else "online"
    run = wandb.init(
        project="io-chess-engine",
        name=f"newtest_phase{args.phase}_{args.branch_dim}bd_{args.mixer_out}m",
        config={
            "phase": args.phase,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "branch_dim": args.branch_dim,
            "mixer_out": args.mixer_out,
            "expert_bottleneck": args.expert_bottleneck,
            "expert_hidden": args.expert_hidden,
            "expert_pool": args.expert_pool,
        },
        mode=mode,
        tags=[phase_name, "model"],
    )
    print(f"{C.GREEN}[wandb] Initialized in {mode} mode{C.RESET}")
    return run


def build_loader(dataset, batch_size, shuffle, workers, custom_sampler=None):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    if custom_sampler is not None:
        kwargs["sampler"] = custom_sampler
    elif shuffle:
        if isinstance(dataset, ConcatDataset):
            kwargs["sampler"] = ConcatChunkedRandomSampler(
                dataset.datasets,
                chunk_size=1_000_000,
            )
        else:
            kwargs["sampler"] = ChunkedRandomSampler(dataset, chunk_size=1_000_000)
    else:
        kwargs["shuffle"] = False
    return DataLoader(dataset, **kwargs)


def parse_data_roots(args):
    roots = []

    if args.data_dirs:
        roots.extend([p.strip() for p in args.data_dirs.split(",") if p.strip()])
    if args.data_dir:
        roots.append(args.data_dir.strip())

    # Keep stable order while removing duplicates.
    ordered_unique = []
    seen = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            ordered_unique.append(root)

    if not ordered_unique:
        raise ValueError("Provide --data_dir or --data_dirs")

    for root in ordered_unique:
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

    return ordered_unique


def resolve_data_splits(data_roots):
    split_dirs = []
    for root in data_roots:
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            split_dirs.append((train_dir, val_dir))
        else:
            print(
                f"{C.YELLOW}[WARN] No train/val split in {root}, "
                f"using same directory for both{C.RESET}"
            )
            split_dirs.append((root, root))
    return split_dirs


def build_phase14_datasets(data_roots, n_globals):
    split_dirs = resolve_data_splits(data_roots)

    train_parts = [
        ChessMoEFactorizedDataset(train_dir, n_globals=n_globals)
        for train_dir, _ in split_dirs
    ]
    val_parts = [
        ChessMoEFactorizedDataset(val_dir, n_globals=n_globals)
        for _, val_dir in split_dirs
    ]

    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    if len(train_parts) > 1:
        print(f"{C.CYAN}Using {len(train_parts)} dataset roots via ConcatDataset.{C.RESET}")

    return train_ds, val_ds


def phase_eta_min(phase):
    # Phase 1 tolerates a slightly higher floor; later fine-tuning phases use a lower floor.
    return 1e-5 if phase == 1 else 1e-6


def build_scheduler(optimizer, total_steps, warmup_steps, eta_min):
    if total_steps <= 0:
        return None

    # Clamp warmup to [0, total_steps - 1] so cosine always has at least one step.
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps == 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=eta_min,
        )

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=eta_min,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def split_steps_for_epoch(total_steps, val_splits):
    val_splits = max(1, int(val_splits))
    base = total_steps // val_splits
    rem = total_steps % val_splits
    # Distribute remainder so all loader steps are consumed each epoch.
    return [base + (1 if i < rem else 0) for i in range(val_splits)]


def arch_string(args):
    return (
        f"bd{args.branch_dim}_m{args.mixer_out}_"
        f"eb{args.expert_bottleneck}_eh{args.expert_hidden}_{args.expert_pool}"
    )


def phase1_export_state_dict(model):
    """Build a checkpoint state where expert0 is copied to all experts."""
    state = copy.deepcopy(model.state_dict())
    keys = [k for k in state.keys() if k.startswith("experts.0.")]
    for key in keys:
        suffix = key[len("experts.0.") :]
        for expert_idx in range(1, len(model.experts)):
            dst = f"experts.{expert_idx}.{suffix}"
            if dst in state:
                state[dst] = state[key].clone()
    return state


def save_model_checkpoint(path, model, phase, epoch, val_terms, args, extra=None, state_dict_override=None):
    payload = {
        "model": state_dict_override if state_dict_override is not None else model.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "val_loss": val_terms["total"],
        "config": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def empty_loss_terms():
    return {"total": 0.0, "wdl": 0.0}


def add_loss_terms(sums, total_t, wdl_t, n):
    sums["total"] += total_t.item() * n
    sums["wdl"] += wdl_t.item() * n


def finalize_loss_terms(sums, total_samples):
    if total_samples == 0:
        return {"total": float("inf"), "wdl": float("inf")}
    inv = 1.0 / float(total_samples)
    return {
        "total": sums["total"] * inv,
        "wdl": sums["wdl"] * inv,
    }


def format_loss_terms(tag, terms, color):
    return f"{tag}: {color}{terms['total']:.5f}{C.RESET}"


def to_planes_list(branches):
    # Legacy loader shape: [B, 12, 10, 8, 8]
    if branches.dim() == 5:
        return [
            branches[:, i, :ch, :, :]
            for i, ch in enumerate(ChessNetFactorizedMoE.PLANES_PER_TYPE)
        ]

    # Compact loader shape: [B, 54, 8, 8] where channels are concatenated per group.
    if branches.dim() == 4:
        planes_list = []
        offset = 0
        for ch in ChessNetFactorizedMoE.PLANES_PER_TYPE:
            planes_list.append(branches[:, offset : offset + ch, :, :])
            offset += ch
        return planes_list

    raise ValueError(f"Unexpected branch tensor shape: {tuple(branches.shape)}")


def top2_sparse_weights(base_weights):
    top_vals, top_idx = torch.topk(base_weights, k=2, dim=1)
    top_prob = top_vals / (top_vals.sum(dim=1, keepdim=True) + 1e-8)
    sparse = torch.zeros_like(base_weights)
    sparse.scatter_(1, top_idx, top_prob)
    return sparse, top_idx


def unpack_common_batch(batch, device):
    branches = batch["branches"].to(device, non_blocking=True)
    bypass = batch["bypass"].to(device, non_blocking=True)
    global_feats = batch["global"].to(device, non_blocking=True)
    target_wdl = batch["wdl"].to(device, non_blocking=True)
    planes_list = to_planes_list(branches)
    return planes_list, bypass, global_feats, target_wdl


def forward_backbone(model, planes_list, bypass, global_feats):
    branch_outs = []
    for i, branch in enumerate(model.branches):
        branch_outs.append(branch(planes_list[i]))

    x = torch.cat(branch_outs + [bypass], dim=1)
    x = model.pointwise_mixer(x)
    g = model.stem_global(global_feats).view(-1, x.shape[1], 1, 1)
    x = model.mixer_act(x + g)
    return x


def train_phase1_steps(model, data_iter, num_steps, optimizer, scheduler, device):
    model.train()

    loss_sums = empty_loss_terms()
    total_samples = 0

    pbar = tqdm(range(num_steps), desc="Train Split", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")

    for _ in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)

        optimizer.zero_grad()

        # Phase 1 uses a single expert path (expert0) without routing.
        features = forward_backbone(model, planes_list, bypass, global_feats)
        pred_wdl = model.experts[0](features)
        loss_total, loss_wdl = base_loss_components(
            pred_wdl, target_wdl
        )
        if torch.isnan(loss_total):
            continue

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = target_wdl.size(0)
        total_samples += n
        add_loss_terms(loss_sums, loss_total, loss_wdl, n)

        pbar.set_postfix(
            total=f"{loss_total.item():.4f}",
        )

    return finalize_loss_terms(loss_sums, total_samples)


def validate_phase1(model, loader, device):
    model.eval()

    loss_sums = empty_loss_terms()
    total_samples = 0

    pbar = tqdm(
        loader,
        desc="Val",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}",
        mininterval=0.5,
    )

    with torch.no_grad():
        for batch in pbar:
            planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)
            features = forward_backbone(model, planes_list, bypass, global_feats)
            pred_wdl = model.experts[0](features)
            loss_total, loss_wdl = base_loss_components(
                pred_wdl, target_wdl
            )

            n = target_wdl.size(0)
            total_samples += n
            add_loss_terms(loss_sums, loss_total, loss_wdl, n)

    return finalize_loss_terms(loss_sums, total_samples)


def copy_expert0_to_all(model):
    if len(model.experts) <= 1:
        return
    src_state = copy.deepcopy(model.experts[0].state_dict())
    for expert_idx in range(1, len(model.experts)):
        model.experts[expert_idx].load_state_dict(src_state)


def run_phase1(model, args, wandb_run=None):
    print_header("PHASE 1: SINGLE-EXPERT PRETRAIN (NO ROUTING)")

    device = get_device()

    train_ds, val_ds = build_phase14_datasets(args.data_roots, args.n_globals)

    train_loader = build_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = build_loader(val_ds, args.batch_size, False, args.workers)

    print(f"Train: {C.CYAN}{len(train_ds):,}{C.RESET} samples")
    print(f"Val:   {C.CYAN}{len(val_ds):,}{C.RESET} samples")

    model = model.to(device)

    # Phase 1 trains shared backbone + expert0 only.
    for param in model.parameters():
        param.requires_grad = True
    for expert_idx in range(1, len(model.experts)):
        for param in model.experts[expert_idx].parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        eta_min=phase_eta_min(1),
    )
    print(
        f"Scheduler: warmup_steps={min(args.warmup_steps, max(0, total_steps - 1))}, "
        f"total_steps={total_steps}, eta_min={phase_eta_min(1):.1e}"
    )

    best_val = float("inf")
    best_state = None

    tag = arch_string(args)
    phase1_best_val_path = os.path.join(args.checkpoint_dir, f"model_{tag}_phase1_best_val.pt")
    phase1_final_path = os.path.join(args.checkpoint_dir, f"model_{tag}_phase1_final.pt")
    phase1_legacy_path = os.path.join(args.checkpoint_dir, f"model_{tag}_phase1.pt")

    split_steps = split_steps_for_epoch(len(train_loader), args.val_splits)

    for epoch in range(args.epochs):
        data_iter = iter(train_loader)

        for split_idx, num_steps in enumerate(split_steps):
            if num_steps <= 0:
                continue

            t0 = time.time()

            train_terms = train_phase1_steps(
                model,
                data_iter,
                num_steps,
                optimizer,
                scheduler,
                device,
            )
            val_terms = validate_phase1(
                model,
                val_loader,
                device,
            )

            train_loss = train_terms["total"]
            val_loss = val_terms["total"]

            elapsed = time.time() - t0
            is_best = val_loss < best_val

            phase1_state = phase1_export_state_dict(model)

            epoch_progress = epoch + (split_idx + 1) / max(1, args.val_splits)

            if is_best:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                save_model_checkpoint(
                    phase1_best_val_path,
                    model,
                    phase=1,
                    epoch=epoch_progress,
                    val_terms=val_terms,
                    args=args,
                    extra={
                        "phase1_single_expert_init": True,
                        "best_metric": "val_total",
                    },
                    state_dict_override=phase1_state,
                )
                # Keep legacy naming for downstream scripts that expect this file.
                save_model_checkpoint(
                    phase1_legacy_path,
                    model,
                    phase=1,
                    epoch=epoch_progress,
                    val_terms=val_terms,
                    args=args,
                    extra={
                        "phase1_single_expert_init": True,
                        "best_metric": "val_total",
                    },
                    state_dict_override=phase1_state,
                )

            save_model_checkpoint(
                phase1_final_path,
                model,
                phase=1,
                epoch=epoch_progress,
                val_terms=val_terms,
                args=args,
                extra={
                    "phase1_single_expert_init": True,
                    "best_metric": "final",
                },
                state_dict_override=phase1_state,
            )

            lr = optimizer.param_groups[0]["lr"]
            marks = []
            if is_best:
                marks.append(f"{C.GREEN}* BEST{C.RESET}")
            best_str = f" {' '.join(marks)}" if marks else ""
            print(
                f"{C.YELLOW}Ep {epoch + 1}/{args.epochs} "
                f"(Split {split_idx + 1}/{args.val_splits}){C.RESET} | "
                f"{format_loss_terms('train', train_terms, C.CYAN)} | "
                f"{format_loss_terms('val', val_terms, C.MAGENTA)} | "
                f"lr: {lr:.2e} | {C.GREEN}{format_time(elapsed)}{C.RESET}{best_str}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": epoch + 1 + (split_idx / max(1, args.val_splits)),
                        "phase": 1,
                        "split": split_idx + 1,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "lr": lr,
                    }
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    copy_expert0_to_all(model)

    print(f"\n{C.GREEN}Phase 1 complete!{C.RESET}")
    print(f"  Best val:     {phase1_best_val_path}")
    print(f"  Final:        {phase1_final_path}")
    print(f"  Legacy phase1:{phase1_legacy_path}")
    return model, phase1_legacy_path


def train_expert_steps(model, expert_idx, data_iter, num_steps, optimizer, scheduler, device):
    model.train()

    # Keep shared backbone in eval mode while experts are specialized.
    model.pointwise_mixer.eval()
    model.stem_global.eval()
    for branch in model.branches:
        branch.eval()

    loss_sums = empty_loss_terms()
    total_samples = 0

    pbar = tqdm(range(num_steps), desc=f"Expert {expert_idx} Split", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")

    for _ in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = forward_backbone(model, planes_list, bypass, global_feats)

        pred_wdl = model.experts[expert_idx](features)
        loss_total, loss_wdl = base_loss_components(
            pred_wdl, target_wdl
        )

        if torch.isnan(loss_total):
            continue

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.experts[expert_idx].parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = target_wdl.size(0)
        total_samples += n
        add_loss_terms(loss_sums, loss_total, loss_wdl, n)

        pbar.set_postfix(
            total=f"{loss_total.item():.4f}",
        )

    return finalize_loss_terms(loss_sums, total_samples)


def validate_expert(model, expert_idx, loader, device):
    model.eval()
    loss_sums = empty_loss_terms()
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Val E{expert_idx}",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}",
        mininterval=0.5,
    )

    with torch.no_grad():
        for batch in pbar:
            planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)
            features = forward_backbone(model, planes_list, bypass, global_feats)
            pred_wdl = model.experts[expert_idx](features)
            loss_total, loss_wdl = base_loss_components(
                pred_wdl, target_wdl
            )

            n = target_wdl.size(0)
            total_samples += n
            add_loss_terms(loss_sums, loss_total, loss_wdl, n)

    return finalize_loss_terms(loss_sums, total_samples)


def load_checkpoint_into_model(model, checkpoint_path, device):
    if not checkpoint_path:
        raise ValueError("checkpoint path is required")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    allowed_unexpected = {"gating.weight", "gating.bias"}
    unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint/state_dict mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
        )
    return ckpt


def run_phase2(model, args, wandb_run=None):
    print_header("PHASE 2: EXPERT SPECIALIZATION")

    device = get_device()
    model = model.to(device)
    ckpt = load_checkpoint_into_model(model, args.checkpoint, device)
    if isinstance(ckpt, dict) and ckpt.get("phase1_single_expert_init", False):
        print(f"{C.GREEN}Phase-1 single-expert initialization detected (expert0 copied to all experts).{C.RESET}")
    elif isinstance(ckpt, dict) and ckpt.get("phase") == 1:
        print(
            f"{C.YELLOW}[WARN] Phase-1 checkpoint has no explicit single-expert init flag; "
            f"specialization may start from non-identical expert heads.{C.RESET}"
        )

    split_dirs = resolve_data_splits(args.data_roots)

    n_experts = min(args.n_experts, 4)
    expert_names = ["Tactical", "Strategic", "MajorEnd", "MinorEnd"]
    all_results = []
    tag = arch_string(args)

    for expert_idx in range(n_experts):
        name = expert_names[expert_idx]
        print(f"\n{C.YELLOW}{'-' * 70}{C.RESET}")
        print(f"{C.YELLOW}Training Expert {expert_idx}: {name}{C.RESET}")
        print(f"{C.YELLOW}{'-' * 70}{C.RESET}")

        train_parts = [
            ChessExpertFactorizedDataset(train_dir, expert_idx, n_globals=args.n_globals)
            for train_dir, _ in split_dirs
        ]
        val_parts = [
            ChessExpertFactorizedDataset(val_dir, expert_idx, n_globals=args.n_globals)
            for _, val_dir in split_dirs
        ]

        train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

        if len(train_parts) == 1:
            train_parts[0]._lazy_load()
            if train_parts[0].mode == "indices":
                expert_sampler = ExpertChunkedSampler(
                    train_parts[0].indices,
                    chunk_size=1_000_000,
                )
                train_loader = build_loader(
                    train_ds,
                    args.batch_size,
                    False,
                    args.workers,
                    custom_sampler=expert_sampler,
                )
            else:
                train_loader = build_loader(train_ds, args.batch_size, True, args.workers)
        else:
            all_index_mode = True
            for part in train_parts:
                part._lazy_load()
                if part.mode != "indices":
                    all_index_mode = False

            if all_index_mode:
                expert_sampler = ConcatExpertChunkedSampler(
                    train_parts,
                    chunk_size=1_000_000,
                )
                train_loader = build_loader(
                    train_ds,
                    args.batch_size,
                    False,
                    args.workers,
                    custom_sampler=expert_sampler,
                )
            else:
                train_loader = build_loader(train_ds, args.batch_size, True, args.workers)
        val_loader = build_loader(val_ds, args.batch_size, False, args.workers)

        print(f"Train: {C.CYAN}{len(train_ds):,}{C.RESET} samples")
        print(f"Val:   {C.CYAN}{len(val_ds):,}{C.RESET} samples")

        for param in model.parameters():
            param.requires_grad = False
        for param in model.experts[expert_idx].parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {C.CYAN}{trainable:,}{C.RESET}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
        total_steps = len(train_loader) * args.epochs
        scheduler = build_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            eta_min=phase_eta_min(2),
        )
        print(
            f"Scheduler: warmup_steps={min(args.warmup_steps, max(0, total_steps - 1))}, "
            f"total_steps={total_steps}, eta_min={phase_eta_min(2):.1e}"
        )

        best_val = float("inf")
        best_state = None
        history = []

        phase2_best_val_path = os.path.join(
            args.checkpoint_dir, f"model_{tag}_phase2_expert{expert_idx}_best_val.pt"
        )
        phase2_final_path = os.path.join(
            args.checkpoint_dir, f"model_{tag}_phase2_expert{expert_idx}_final.pt"
        )

        split_steps = split_steps_for_epoch(len(train_loader), args.val_splits)

        for epoch in range(args.epochs):
            data_iter = iter(train_loader)

            for split_idx, num_steps in enumerate(split_steps):
                if num_steps <= 0:
                    continue

                t0 = time.time()
                train_terms = train_expert_steps(
                    model,
                    expert_idx,
                    data_iter,
                    num_steps,
                    optimizer,
                    scheduler,
                    device,
                )
                val_terms = validate_expert(model, expert_idx, val_loader, device)

                train_loss = train_terms["total"]
                val_loss = val_terms["total"]

                elapsed = time.time() - t0
                is_best = val_loss < best_val
                epoch_progress = epoch + (split_idx + 1) / max(1, args.val_splits)

                if is_best:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.experts[expert_idx].state_dict())
                    save_model_checkpoint(
                        phase2_best_val_path,
                        model,
                        phase=2,
                        epoch=epoch_progress,
                        val_terms=val_terms,
                        args=args,
                        extra={
                            "expert_idx": expert_idx,
                            "expert_name": name,
                            "best_metric": "val_total",
                        },
                    )

                save_model_checkpoint(
                    phase2_final_path,
                    model,
                    phase=2,
                    epoch=epoch_progress,
                    val_terms=val_terms,
                    args=args,
                    extra={
                        "expert_idx": expert_idx,
                        "expert_name": name,
                        "best_metric": "final",
                    },
                )

                lr = optimizer.param_groups[0]["lr"]
                marks = []
                if is_best:
                    marks.append(f"{C.GREEN}* BEST{C.RESET}")
                best_str = f" {' '.join(marks)}" if marks else ""
                print(
                    f"  {C.YELLOW}Ep {epoch + 1}/{args.epochs} "
                    f"(Split {split_idx + 1}/{args.val_splits}){C.RESET} | "
                    f"{format_loss_terms('train', train_terms, C.CYAN)} | "
                    f"{format_loss_terms('val', val_terms, C.MAGENTA)} | "
                    f"lr: {lr:.2e} | {C.GREEN}{format_time(elapsed)}{C.RESET}{best_str}"
                )

                history.append(
                    {
                        "epoch": epoch + 1 + (split_idx / max(1, args.val_splits)),
                        "split": split_idx + 1,
                        "train_loss": train_terms["total"],
                        "val_loss": val_terms["total"],
                    }
                )

                if wandb_run:
                    wandb_run.log(
                        {
                            "phase": 2,
                            "expert_idx": expert_idx,
                            "epoch": epoch + 1 + (split_idx / max(1, args.val_splits)),
                            "split": split_idx + 1,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "lr": lr,
                        }
                    )

        if best_state is not None:
            model.experts[expert_idx].load_state_dict(best_state)

        all_results.append(
            {
                "expert": name,
                "best_val_loss": best_val,
                "best_val_path": phase2_best_val_path,
                "final_path": phase2_final_path,
                "history": history,
            }
        )

        print(f"  {C.GREEN}* Best total: {best_val:.5f}{C.RESET}")

    save_path = os.path.join(args.checkpoint_dir, f"model_{tag}_phase2.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "phase": 2,
            "results": all_results,
            "config": vars(args),
        },
        save_path,
    )

    print(f"\n{C.GREEN}Phase 2 complete!{C.RESET} Checkpoint: {save_path}")
    return model, save_path


def train_phase4_steps(model, data_iter, num_steps, optimizer, scheduler, device):
    model.train()
    loss_sums = empty_loss_terms()
    total_samples = 0
    top2_usage = torch.zeros(4, device=device)

    pbar = tqdm(range(num_steps), desc="Train Split", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")

    for _ in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)
        base_weights = batch["base_weights"].to(device, non_blocking=True)

        route_weights, top2_idx = top2_sparse_weights(base_weights)

        optimizer.zero_grad()

        pred_wdl = model(planes_list, bypass, global_feats, weights=route_weights)
        loss_total, loss_wdl = base_loss_components(
            pred_wdl, target_wdl
        )
        if torch.isnan(loss_total):
            continue

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = target_wdl.size(0)
        total_samples += n
        add_loss_terms(loss_sums, loss_total, loss_wdl, n)

        for i in range(4):
            top2_usage[i] += (top2_idx == i).sum()

        pbar.set_postfix(
            total=f"{loss_total.item():.4f}",
        )

    if total_samples == 0:
        return finalize_loss_terms(loss_sums, total_samples), torch.zeros(4)

    usage_pct = (top2_usage / (total_samples * 2) * 100.0).detach().cpu()
    return finalize_loss_terms(loss_sums, total_samples), usage_pct


def validate_phase4(model, loader, device):
    model.eval()
    loss_sums = empty_loss_terms()
    total_samples = 0

    pbar = tqdm(
        loader,
        desc="Val",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}",
        mininterval=0.5,
    )

    with torch.no_grad():
        for batch in pbar:
            planes_list, bypass, global_feats, target_wdl = unpack_common_batch(batch, device)
            base_weights = batch["base_weights"].to(device, non_blocking=True)
            route_weights, _ = top2_sparse_weights(base_weights)

            pred_wdl = model(planes_list, bypass, global_feats, weights=route_weights)
            loss_total, loss_wdl = base_loss_components(
                pred_wdl, target_wdl
            )

            n = target_wdl.size(0)
            total_samples += n
            add_loss_terms(loss_sums, loss_total, loss_wdl, n)

    return finalize_loss_terms(loss_sums, total_samples)


def run_phase4(model, args, wandb_run=None):
    print_header("PHASE 4: JOINT FINE-TUNING (TOP-2)")

    device = get_device()
    model = model.to(device)
    load_checkpoint_into_model(model, args.checkpoint, device)

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All params trainable: {C.CYAN}{trainable:,}{C.RESET}")

    train_ds, val_ds = build_phase14_datasets(args.data_roots, args.n_globals)

    train_loader = build_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = build_loader(val_ds, args.batch_size, False, args.workers)

    lr = args.lr * 0.1
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        eta_min=phase_eta_min(4),
    )

    print(f"LR: {C.CYAN}{lr:.2e}{C.RESET} (0.1x base)")
    print(
        f"Scheduler: warmup_steps={min(args.warmup_steps, max(0, total_steps - 1))}, "
        f"total_steps={total_steps}, eta_min={phase_eta_min(4):.1e}"
    )
    print(f"Train: {C.CYAN}{len(train_ds):,}{C.RESET} samples")
    print(f"Val:   {C.CYAN}{len(val_ds):,}{C.RESET} samples")

    best_val = float("inf")
    best_state = None
    tag = arch_string(args)
    expert_names = ["Tac", "Str", "Maj", "Min"]

    best_val_path = os.path.join(args.checkpoint_dir, f"model_{tag}_best.pt")
    final_last_path = os.path.join(args.checkpoint_dir, f"model_{tag}_last.pt")
    final_path = os.path.join(args.checkpoint_dir, f"model_{tag}_final.pt")

    split_steps = split_steps_for_epoch(len(train_loader), args.val_splits)

    for epoch in range(args.epochs):
        data_iter = iter(train_loader)

        for split_idx, num_steps in enumerate(split_steps):
            if num_steps <= 0:
                continue

            t0 = time.time()

            train_terms, usage = train_phase4_steps(
                model,
                data_iter,
                num_steps,
                optimizer,
                scheduler,
                device,
            )
            val_terms = validate_phase4(
                model,
                val_loader,
                device,
            )

            train_loss = train_terms["total"]
            val_loss = val_terms["total"]

            elapsed = time.time() - t0
            is_best = val_loss < best_val
            epoch_progress = epoch + (split_idx + 1) / max(1, args.val_splits)
            if is_best:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                save_model_checkpoint(
                    best_val_path,
                    model,
                    phase=4,
                    epoch=epoch_progress,
                    val_terms=val_terms,
                    args=args,
                    extra={"best_metric": "val_total"},
                )

            lr_now = optimizer.param_groups[0]["lr"]
            usage_str = " | ".join([f"{expert_names[i]}:{usage[i]:.0f}%" for i in range(4)])
            marks = []
            if is_best:
                marks.append(f"{C.GREEN}* BEST{C.RESET}")
            best_str = f" {' '.join(marks)}" if marks else ""

            print(
                f"{C.YELLOW}Ep {epoch + 1}/{args.epochs} "
                f"(Split {split_idx + 1}/{args.val_splits}){C.RESET} | "
                f"{format_loss_terms('train', train_terms, C.CYAN)} | "
                f"{format_loss_terms('val', val_terms, C.MAGENTA)} | "
                f"lr: {lr_now:.2e} | {C.GREEN}{format_time(elapsed)}{C.RESET} | "
                f"[{usage_str}]{best_str}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": epoch + 1 + (split_idx / max(1, args.val_splits)),
                        "phase": 4,
                        "split": split_idx + 1,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "lr": lr_now,
                        "usage/tactical": usage[0].item(),
                        "usage/strategic": usage[1].item(),
                        "usage/major": usage[2].item(),
                        "usage/minor": usage[3].item(),
                    }
                )

            save_model_checkpoint(
                final_last_path,
                model,
                phase=4,
                epoch=epoch_progress,
                val_terms=val_terms,
                args=args,
                extra={"best_metric": "final"},
            )
            save_model_checkpoint(
                final_path,
                model,
                phase=4,
                epoch=epoch_progress,
                val_terms=val_terms,
                args=args,
                extra={"best_metric": "final"},
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n{C.GREEN}Phase 4 complete!{C.RESET} Best val loss: {best_val:.5f}")
    print(f"Best val checkpoint:     {best_val_path}")
    print(f"Final checkpoint:        {final_path}")
    return model, best_val_path


def main():
    parser = argparse.ArgumentParser(description="Train model (factorized MoE)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 4])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--data_dirs",
        type=str,
        default=None,
        help="Comma-separated dataset roots to concatenate (e.g. nvme_root,usb_root)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Steps for linear warmup")
    parser.add_argument("--val_splits", type=int, default=4, help="Number of validations per epoch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_offline", action="store_true", help="W&B offline mode")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_globals", type=int, default=21)
    parser.add_argument("--branch_dim", type=int, default=16)
    parser.add_argument("--mixer_out", type=int, default=64)
    parser.add_argument("--n_bypass", type=int, default=12)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--expert_bottleneck", type=int, default=32)
    parser.add_argument("--expert_hidden", type=int, default=128)
    parser.add_argument(
        "--expert_pool",
        type=str,
        default="flat",
        choices=["flat", "gap", "pool2avg", "pool2max"],
    )
    args = parser.parse_args()
    args.data_roots = parse_data_roots(args)

    if args.n_experts != 4:
        raise ValueError(
            "This training pipeline expects n_experts=4 because dataset expert_weights.bin "
            "stores 4 base expert routing targets."
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    print(f"\n{C.CYAN}Device:{C.RESET} {device}")
    print(
        f"{C.CYAN}Architecture:{C.RESET} "
        f"branch_dim={args.branch_dim}, mixer_out={args.mixer_out}, "
        f"experts={args.n_experts}, pool={args.expert_pool}"
    )
    print(f"{C.CYAN}Dataset roots:{C.RESET} {', '.join(args.data_roots)}")

    model = ChessNetFactorizedMoE(
        n_globals=args.n_globals,
        branch_dim=args.branch_dim,
        mixer_out=args.mixer_out,
        n_bypass=args.n_bypass,
        n_experts=args.n_experts,
        expert_bottleneck=args.expert_bottleneck,
        expert_hidden=args.expert_hidden,
        expert_pool=args.expert_pool,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{C.CYAN}Total params:{C.RESET} {total_params:,}")

    phase_names = {1: "base", 2: "experts", 4: "finetune"}
    wandb_run = init_wandb(args, phase_names[args.phase])

    if args.phase == 1:
        run_phase1(model, args, wandb_run)
    elif args.phase == 2:
        if not args.checkpoint:
            raise ValueError("Phase 2 requires --checkpoint from phase 1")
        run_phase2(model, args, wandb_run)
    elif args.phase == 4:
        if not args.checkpoint:
            raise ValueError("Phase 4 requires --checkpoint from phase 2")
        run_phase4(model, args, wandb_run)

    if wandb_run:
        wandb_run.finish()

    print(f"\n{C.GREEN}{'=' * 70}{C.RESET}")
    print(f"{C.GREEN}  Done!{C.RESET}")
    print(f"{C.GREEN}{'=' * 70}{C.RESET}\n")


if __name__ == "__main__":
    main()
