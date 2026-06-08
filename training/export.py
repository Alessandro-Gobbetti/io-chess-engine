#!/usr/bin/env python3
"""
@file export.py
@brief Export parity artifacts for training/model.py

1) Native float32 weights for a C++ loader
2) ONNX models (monolithic + split backbone/experts)
3) Deterministic input samples and PyTorch reference outputs
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from model import ChessNetFactorizedMoE


MAGIC_WEIGHTS = 0x32454F4D  # "MOE2"
MAGIC_INPUTS = 0x32504E49  # "INP2"
MAGIC_REFS = 0x32464552  # "REF2"
VERSION = 1
MAX_BRANCH_PLANES = 10
POOL_TO_CODE = {
    "flat": 0,
    "gap": 1,
    "pool2avg": 2,
    "pool2max": 3,
}


def _extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            sd = ckpt_obj["model"]
        elif "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            sd = ckpt_obj["state_dict"]
        else:
            sd = ckpt_obj
    else:
        raise ValueError("checkpoint must be a dict/state_dict")

    def strip_prefix_if_present(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        keys = list(d.keys())
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in d.items()}
        return d

    sd = strip_prefix_if_present(sd, "module.")
    sd = strip_prefix_if_present(sd, "model.")
    return sd


def _write_u32s(f, values):
    f.write(struct.pack("<" + "I" * len(values), *values))


def _write_f32_array(f, arr: np.ndarray):
    np.asarray(arr, dtype=np.float32, order="C").tofile(f)


class OnnxWrapper(nn.Module):
    def __init__(self, model: ChessNetFactorizedMoE):
        super().__init__()
        self.model = model
        self.planes_per_type = list(model.PLANES_PER_TYPE)

    def forward(self, branch_in: torch.Tensor, bypass_in: torch.Tensor, global_in: torch.Tensor):
        planes_list = []
        for i, ch in enumerate(self.planes_per_type):
            planes_list.append(branch_in[:, i, :ch, :, :])
        wdl = self.model(planes_list, bypass_in, global_in)
        return wdl


class OnnxBackboneWrapper(nn.Module):
    def __init__(self, model: ChessNetFactorizedMoE):
        super().__init__()
        self.model = model
        self.planes_per_type = list(model.PLANES_PER_TYPE)

    def forward(self, branch_in: torch.Tensor, bypass_in: torch.Tensor, global_in: torch.Tensor):
        planes_list = []
        for i, ch in enumerate(self.planes_per_type):
            planes_list.append(branch_in[:, i, :ch, :, :])

        branch_outs = []
        for i, branch in enumerate(self.model.branches):
            branch_outs.append(branch(planes_list[i]))

        x = torch.cat(branch_outs + [bypass_in], dim=1)
        x = self.model.pointwise_mixer(x)
        g = self.model.stem_global(global_in).view(-1, x.shape[1], 1, 1)
        x = self.model.mixer_act(x + g)
        return x


class OnnxExpertWrapper(nn.Module):
    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert

    def forward(self, features: torch.Tensor):
        wdl = self.expert(features)
        return wdl


def export_native_weights(out_path: Path, model: ChessNetFactorizedMoE) -> None:
    branch_dim = model.branches[0].conv0.out_channels
    mixer_out = model.pointwise_mixer.out_channels
    n_bypass = model.pointwise_mixer.in_channels - 12 * branch_dim
    n_globals = model.stem_global.in_features
    n_experts = model.n_experts
    expert_bottleneck = model.experts[0].head_conv.out_channels
    expert_hidden = model.experts[0].head_hidden.out_features
    pool_code = POOL_TO_CODE[model.expert_pool]

    with out_path.open("wb") as f:
        _write_u32s(
            f,
            [
                MAGIC_WEIGHTS,
                VERSION,
                branch_dim,
                mixer_out,
                n_bypass,
                n_globals,
                n_experts,
                expert_bottleneck,
                expert_hidden,
                pool_code,
                12,
                MAX_BRANCH_PLANES,
            ],
        )
        _write_u32s(f, list(model.PLANES_PER_TYPE))

        for b in range(12):
            branch = model.branches[b]
            conv0_w = branch.conv0.weight.detach().cpu().numpy().astype(np.float32)
            conv0_b = branch.conv0.bias.detach().cpu().numpy().astype(np.float32)
            conv1_w = branch.conv1.weight.detach().cpu().numpy().astype(np.float32)
            conv1_b = branch.conv1.bias.detach().cpu().numpy().astype(np.float32)
            conv2_w = (
                branch.conv2.weight.detach().cpu().numpy().astype(np.float32).reshape(branch_dim, branch_dim)
            )
            conv2_b = branch.conv2.bias.detach().cpu().numpy().astype(np.float32)

            _write_f32_array(f, conv0_w)
            _write_f32_array(f, conv0_b)
            _write_f32_array(f, conv1_w)
            _write_f32_array(f, conv1_b)
            _write_f32_array(f, conv2_w)
            _write_f32_array(f, conv2_b)

        mixer_w = (
            model.pointwise_mixer.weight.detach().cpu().numpy().astype(np.float32).reshape(mixer_out, 12 * branch_dim + n_bypass)
        )
        mixer_b = model.pointwise_mixer.bias.detach().cpu().numpy().astype(np.float32)
        stem_w = model.stem_global.weight.detach().cpu().numpy().astype(np.float32)
        stem_b = model.stem_global.bias.detach().cpu().numpy().astype(np.float32)
        # Kept in file format for backward compatibility with native loader.
        # Routing is no longer learned in Python training, so these are zeros.
        gate_w = np.zeros((n_experts, n_globals), dtype=np.float32)
        gate_b = np.zeros((n_experts,), dtype=np.float32)

        _write_f32_array(f, mixer_w)
        _write_f32_array(f, mixer_b)
        _write_f32_array(f, stem_w)
        _write_f32_array(f, stem_b)
        _write_f32_array(f, gate_w)
        _write_f32_array(f, gate_b)

        for ex in model.experts:
            ex_conv_w = (
                ex.head_conv.weight.detach().cpu().numpy().astype(np.float32).reshape(expert_bottleneck, mixer_out)
            )
            ex_conv_b = ex.head_conv.bias.detach().cpu().numpy().astype(np.float32)
            ex_hidden_w = ex.head_hidden.weight.detach().cpu().numpy().astype(np.float32)
            ex_hidden_b = ex.head_hidden.bias.detach().cpu().numpy().astype(np.float32)
            ex_wdl_w = ex.head_wdl.weight.detach().cpu().numpy().astype(np.float32)
            ex_wdl_b = ex.head_wdl.bias.detach().cpu().numpy().astype(np.float32)

            _write_f32_array(f, ex_conv_w)
            _write_f32_array(f, ex_conv_b)
            _write_f32_array(f, ex_hidden_w)
            _write_f32_array(f, ex_hidden_b)
            _write_f32_array(f, ex_wdl_w)
            _write_f32_array(f, ex_wdl_b)


def export_inputs_and_refs(
    inputs_path: Path,
    refs_path: Path,
    model: ChessNetFactorizedMoE,
    n_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    branch_dim = model.branches[0].conv0.out_channels
    n_bypass = model.pointwise_mixer.in_channels - 12 * branch_dim
    n_globals = model.stem_global.in_features

    branches = rng.standard_normal((n_samples, 12, MAX_BRANCH_PLANES, 8, 8), dtype=np.float32)
    bypass = rng.standard_normal((n_samples, n_bypass, 8, 8), dtype=np.float32)
    global_v = rng.standard_normal((n_samples, n_globals), dtype=np.float32)

    refs = np.zeros((n_samples, 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            planes_list = []
            for b, ch in enumerate(model.PLANES_PER_TYPE):
                t = torch.from_numpy(branches[i : i + 1, b, :ch, :, :])
                planes_list.append(t)
            t_bypass = torch.from_numpy(bypass[i : i + 1])
            t_global = torch.from_numpy(global_v[i : i + 1])
            wdl = model(planes_list, t_bypass, t_global)
            refs[i, :3] = wdl.squeeze(0).cpu().numpy().astype(np.float32)

    with inputs_path.open("wb") as f:
        _write_u32s(
            f,
            [
                MAGIC_INPUTS,
                VERSION,
                n_samples,
                n_bypass,
                n_globals,
                MAX_BRANCH_PLANES,
                12,
            ],
        )
        _write_u32s(f, list(model.PLANES_PER_TYPE))
        _write_f32_array(f, branches)
        _write_f32_array(f, bypass)
        _write_f32_array(f, global_v)

    with refs_path.open("wb") as f:
        _write_u32s(f, [MAGIC_REFS, VERSION, n_samples, 3])
        _write_f32_array(f, refs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model parity artifacts")
    parser.add_argument("--out-dir", type=str, default="training/parity_bundle_moe_cache")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n-globals", type=int, default=21)
    parser.add_argument("--branch-dim", type=int, default=16)
    parser.add_argument("--mixer-out", type=int, default=64)
    parser.add_argument("--n-bypass", type=int, default=12)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--expert-bottleneck", type=int, default=32)
    parser.add_argument("--expert-hidden", type=int, default=128)
    parser.add_argument("--expert-pool", type=str, default="flat", choices=["flat", "gap", "pool2avg", "pool2max"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

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

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)
        incompatible = model.load_state_dict(sd, strict=False)
        allowed_unexpected = {"gating.weight", "gating.bias"}
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Checkpoint/state_dict mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
            )

    model.eval()

    weights_path = out_dir / "native_weights.bin"
    onnx_path = out_dir / "model.onnx"
    backbone_onnx_path = out_dir / "backbone.onnx"
    inputs_path = out_dir / "inputs.bin"
    refs_path = out_dir / "refs.bin"

    export_native_weights(weights_path, model)

    wrapper = OnnxWrapper(model).eval()
    dummy_branch = torch.randn(1, 12, MAX_BRANCH_PLANES, 8, 8, dtype=torch.float32)
    dummy_bypass = torch.randn(1, args.n_bypass, 8, 8, dtype=torch.float32)
    dummy_global = torch.randn(1, args.n_globals, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_branch, dummy_bypass, dummy_global),
        onnx_path.as_posix(),
        input_names=["branch_in", "bypass_in", "global_in"],
        output_names=["wdl"],
        dynamic_axes={
            "branch_in": {0: "batch"},
            "bypass_in": {0: "batch"},
            "global_in": {0: "batch"},
            "wdl": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    backbone_wrapper = OnnxBackboneWrapper(model).eval()
    torch.onnx.export(
        backbone_wrapper,
        (dummy_branch, dummy_bypass, dummy_global),
        backbone_onnx_path.as_posix(),
        input_names=["branch_in", "bypass_in", "global_in"],
        output_names=["features"],
        dynamic_axes={
            "branch_in": {0: "batch"},
            "bypass_in": {0: "batch"},
            "global_in": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    dummy_features = torch.randn(1, args.mixer_out, 8, 8, dtype=torch.float32)
    for i, expert in enumerate(model.experts):
        expert_wrapper = OnnxExpertWrapper(expert).eval()
        expert_path = out_dir / f"expert{i}.onnx"
        torch.onnx.export(
            expert_wrapper,
            (dummy_features,),
            expert_path.as_posix(),
            input_names=["features"],
            output_names=["wdl"],
            dynamic_axes={
                "features": {0: "batch"},
                "wdl": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    export_inputs_and_refs(inputs_path, refs_path, model, n_samples=args.samples, seed=args.seed + 1)

    print(f"Exported native weights: {weights_path}")
    print(f"Exported ONNX (monolithic): {onnx_path}")
    print(f"Exported ONNX backbone: {backbone_onnx_path}")
    for i in range(len(model.experts)):
        print(f"Exported ONNX expert{i}: {out_dir / f'expert{i}.onnx'}")
    print(f"Exported inputs: {inputs_path}")
    print(f"Exported refs: {refs_path}")


if __name__ == "__main__":
    main()
