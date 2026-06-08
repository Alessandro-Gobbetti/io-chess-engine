"""
@file dataset.py
@brief Dataset utilities for training model.py on packed factorized features.

Loads packed factorized records produced by preprocessing in --factorized mode:
  - features.bin: PackedFactorizedInput
  - labels.bin: WDLOutput (3 floats: win, draw, loss)
  - expert_weights.bin: 6 floats [base0-3, survivor, killer]

PackedFactorizedInput C++ layout (alignas(64)):
    uint8_t branches[54][64]   # only used branch planes (4/5 per group)
  uint8_t bypass[12][64]
  float   global[32]
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


PLANES_PER_GROUP_CONST = [4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4]
PACKED_OFFSETS_CONST = [
    sum(PLANES_PER_GROUP_CONST[:i]) for i in range(len(PLANES_PER_GROUP_CONST))
]

def decode_feature_row(row, feature_layout, planes_per_type, packed_offsets, packed_branch_planes, num_bypass_planes):
    if feature_layout == "compact54":
        raw_branches = np.array(row["branches"], copy=True)
        branches = np.unpackbits(raw_branches.view(np.uint8), bitorder='little').astype(np.float32).reshape(packed_branch_planes, 8, 8)
        
        raw_bypass_cont = np.array(row["bypass_continuous"], copy=True)
        raw_bypass_cat = np.array(row["bypass_categorical"], copy=True)
        
        bypass = np.empty((num_bypass_planes, 8, 8), dtype=np.float32)
        bypass[:2] = raw_bypass_cont.reshape(2, 8, 8).astype(np.float32) / 255.0
        bypass[2:] = np.unpackbits(raw_bypass_cat.view(np.uint8), bitorder='little').astype(np.float32).reshape(10, 8, 8)
        
        return branches, bypass
    else:
        legacy = np.array(row["branches"], copy=True).reshape(12, 10, 8, 8)
        branches = np.empty((packed_branch_planes, 8, 8), dtype=np.float32)
        for g, planes in enumerate(planes_per_type):
            offset = packed_offsets[g]
            branches[offset : offset + planes] = legacy[g, :planes].astype(np.float32) / 255.0
            
        bypass = np.array(row["bypass"], copy=True).reshape(num_bypass_planes, 8, 8).astype(np.float32) / 255.0
        return branches, bypass


class ChessMoEFactorizedDataset(Dataset):
    """
    Dataset for factorized MoE training.

    Returns dict with:
    - branches: [54, 8, 8] uint8 (compact packed branch planes)
      - bypass: [12, 8, 8] uint8 (packed bypass planes)
      - global: [n_globals] float32 (default first 21 globals)
      - wdl: [3] float32
      - base_weights: [4] float32
      - aux_weights: [2] float32
    """

    PLANES_PER_TYPE = PLANES_PER_GROUP_CONST
    MAX_BRANCH_PLANES = 10
    PACKED_BRANCH_PLANES = sum(PLANES_PER_TYPE)
    NUM_BYPASS_PLANES = 12
    PACKED_OFFSETS = PACKED_OFFSETS_CONST

    def __init__(self, data_dir, max_samples=None, in_memory=False, n_globals=21):
        self.in_memory = in_memory
        self.data_dir = data_dir
        self.n_globals = n_globals

        self.features_path = os.path.join(data_dir, "features.bin")
        self.labels_path = os.path.join(data_dir, "labels.bin")
        self.weights_path = os.path.join(data_dir, "expert_weights.bin")

        for path in [self.features_path, self.labels_path, self.weights_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        self.features_dtype_legacy = np.dtype(
            [
                ("branches", np.uint8, (12, self.MAX_BRANCH_PLANES, 64)),
                ("bypass", np.uint8, (self.NUM_BYPASS_PLANES, 64)),
                ("global", np.float32, 32),
            ],
            align=True,
        )
        self.features_dtype_compact = np.dtype(
            [
                ("branches", np.uint64, self.PACKED_BRANCH_PLANES),
                ("bypass_continuous", np.uint8, (2, 64)),
                ("bypass_categorical", np.uint64, 10),
                ("global", np.float32, 32),
            ],
            align=True,
        )

        label_size = 3 * 4
        weight_size = 6 * 4

        label_samples = os.path.getsize(self.labels_path) // label_size
        weight_samples = os.path.getsize(self.weights_path) // weight_size

        if label_samples != weight_samples:
            raise ValueError(
                "Sample count mismatch: "
                f"labels={label_samples}, weights={weight_samples}"
            )

        expected_samples = label_samples
        feature_bytes = os.path.getsize(self.features_path)

        compact_ok = (
            feature_bytes % self.features_dtype_compact.itemsize == 0
            and feature_bytes // self.features_dtype_compact.itemsize == expected_samples
        )
        legacy_ok = (
            feature_bytes % self.features_dtype_legacy.itemsize == 0
            and feature_bytes // self.features_dtype_legacy.itemsize == expected_samples
        )

        if compact_ok:
            self.features_dtype = self.features_dtype_compact
            self.feature_layout = "compact54"
        elif legacy_ok:
            self.features_dtype = self.features_dtype_legacy
            self.feature_layout = "legacy12x10"
        else:
            raise ValueError(
                "Unsupported factorized feature layout: "
                f"features bytes={feature_bytes}, expected samples={expected_samples}, "
                f"compact rec={self.features_dtype_compact.itemsize}, "
                f"legacy rec={self.features_dtype_legacy.itemsize}"
            )

        self.n_samples = expected_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
            print(f"Dataset: Truncated to {self.n_samples:,} samples")
        else:
            print(f"Dataset: {self.n_samples:,} samples ({self.feature_layout})")

        self.features = None
        self.labels = None
        self.weights = None

        if in_memory:
            self._load_to_memory()

    def _load_to_memory(self):
        print("Loading factorized dataset to RAM...")
        self.features = np.fromfile(
            self.features_path, dtype=self.features_dtype, count=self.n_samples
        )
        self.labels = np.fromfile(
            self.labels_path, dtype=np.float32, count=self.n_samples * 3
        ).reshape(-1, 3)
        self.weights = np.fromfile(
            self.weights_path, dtype=np.float32, count=self.n_samples * 6
        ).reshape(-1, 6)
        print("Done.")

    def _lazy_load(self):
        if self.features is None:
            self.features = np.memmap(
                self.features_path,
                dtype=self.features_dtype,
                mode="r",
                shape=(self.n_samples,),
            )
            self.labels = np.memmap(
                self.labels_path, dtype=np.float32, mode="r", shape=(self.n_samples, 3)
            )
            self.weights = np.memmap(
                self.weights_path, dtype=np.float32, mode="r", shape=(self.n_samples, 6)
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._lazy_load()

        row = self.features[idx]
        branches, bypass = decode_feature_row(
            row, self.feature_layout, self.PLANES_PER_TYPE, self.PACKED_OFFSETS,
            self.PACKED_BRANCH_PLANES, self.NUM_BYPASS_PLANES
        )
        global_feats = np.array(row["global"][: self.n_globals], copy=True)

        label = np.array(self.labels[idx], copy=True)
        wdl = label[:3]

        weights = np.array(self.weights[idx], copy=True)
        base_weights = weights[:4]
        aux_weights = weights[4:6]

        return {
            "branches": torch.from_numpy(branches),
            "bypass": torch.from_numpy(bypass),
            "global": torch.from_numpy(global_feats),
            "wdl": torch.from_numpy(wdl),
            "base_weights": torch.from_numpy(base_weights),
            "aux_weights": torch.from_numpy(aux_weights),
        }


class ChessExpertFactorizedDataset(Dataset):
    """
    Expert-specific factorized dataset for phase 2 specialization.

        Preferred format (index-based, no feature duplication):
            - expert{N}_indices.bin
            - features.bin
            - labels.bin

        Legacy fallback format (duplicated expert feature shards):
            - expert{N}_features.bin
            - expert{N}_labels.bin
    """

    EXPERT_NAMES = ["Tactical", "Strategic", "Major End", "Minor End", "Survivor", "Killer"]
    PLANES_PER_TYPE = PLANES_PER_GROUP_CONST
    MAX_BRANCH_PLANES = 10
    PACKED_BRANCH_PLANES = sum(PLANES_PER_TYPE)
    NUM_BYPASS_PLANES = 12
    PACKED_OFFSETS = PACKED_OFFSETS_CONST

    def __init__(self, data_dir, expert_idx, max_samples=None, in_memory=False, n_globals=21):
        self.in_memory = in_memory
        self.data_dir = data_dir
        self.expert_idx = expert_idx
        self.n_globals = n_globals
        self.expert_name = self.EXPERT_NAMES[expert_idx]

        self.indices_path = os.path.join(data_dir, f"expert{expert_idx}_indices.bin")
        self.global_features_path = os.path.join(data_dir, "features.bin")
        self.global_labels_path = os.path.join(data_dir, "labels.bin")

        self.legacy_features_path = os.path.join(data_dir, f"expert{expert_idx}_features.bin")
        self.legacy_labels_path = os.path.join(data_dir, f"expert{expert_idx}_labels.bin")

        self.features_dtype_legacy = np.dtype(
            [
                ("branches", np.uint8, (12, self.MAX_BRANCH_PLANES, 64)),
                ("bypass", np.uint8, (self.NUM_BYPASS_PLANES, 64)),
                ("global", np.float32, 32),
            ],
            align=True,
        )
        self.features_dtype_compact = np.dtype(
            [
                ("branches", np.uint64, self.PACKED_BRANCH_PLANES),
                ("bypass_continuous", np.uint8, (2, 64)),
                ("bypass_categorical", np.uint64, 10),
                ("global", np.float32, 32),
            ],
            align=True,
        )

        self.mode = None
        if (
            os.path.exists(self.indices_path)
            and os.path.exists(self.global_features_path)
            and os.path.exists(self.global_labels_path)
        ):
            self.mode = "indices"
        elif os.path.exists(self.legacy_features_path) and os.path.exists(self.legacy_labels_path):
            self.mode = "legacy"
        else:
            raise FileNotFoundError(
                "Missing expert dataset files. Expected either: "
                f"[{self.indices_path}, {self.global_features_path}, {self.global_labels_path}] "
                "or "
                f"[{self.legacy_features_path}, {self.legacy_labels_path}]"
            )

        self.features = None
        self.labels = None
        self.indices = None

        if self.mode == "indices":
            self._init_index_mode(max_samples)
        else:
            self._init_legacy_mode(max_samples)

        if in_memory:
            self._load_to_memory()

    def _detect_layout(self, feature_path, expected_samples):
        feature_bytes = os.path.getsize(feature_path)
        compact_ok = (
            feature_bytes % self.features_dtype_compact.itemsize == 0
            and feature_bytes // self.features_dtype_compact.itemsize == expected_samples
        )
        legacy_ok = (
            feature_bytes % self.features_dtype_legacy.itemsize == 0
            and feature_bytes // self.features_dtype_legacy.itemsize == expected_samples
        )

        if compact_ok:
            return self.features_dtype_compact, "compact54"
        if legacy_ok:
            return self.features_dtype_legacy, "legacy12x10"

        raise ValueError(
            "Unsupported expert factorized feature layout: "
            f"features bytes={feature_bytes}, expected={expected_samples}, "
            f"compact rec={self.features_dtype_compact.itemsize}, "
            f"legacy rec={self.features_dtype_legacy.itemsize}"
        )

    def _init_index_mode(self, max_samples):
        label_size = 3 * 4
        global_label_samples = os.path.getsize(self.global_labels_path) // label_size
        self.features_dtype, self.feature_layout = self._detect_layout(
            self.global_features_path, global_label_samples
        )
        self.global_n_samples = global_label_samples

        index_bytes = os.path.getsize(self.indices_path)
        if index_bytes % np.dtype(np.uint32).itemsize != 0:
            raise ValueError(
                f"Corrupt index file size for expert {self.expert_idx}: {index_bytes} bytes"
            )

        index_samples = index_bytes // np.dtype(np.uint32).itemsize
        self.n_samples = index_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

        print(
            f"Expert Dataset [{self.expert_idx}] {self.expert_name}: "
            f"{self.n_samples:,} samples (index mode, {self.feature_layout})"
        )

    def _init_legacy_mode(self, max_samples):
        label_size = 3 * 4
        label_samples = os.path.getsize(self.legacy_labels_path) // label_size
        self.features_dtype, self.feature_layout = self._detect_layout(
            self.legacy_features_path, label_samples
        )

        self.n_samples = label_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

        print(
            f"Expert Dataset [{self.expert_idx}] {self.expert_name}: "
            f"{self.n_samples:,} samples (legacy mode, {self.feature_layout})"
        )

    def _load_to_memory(self):
        print(f"Loading expert {self.expert_idx} to RAM...")
        if self.mode == "indices":
            self.indices = np.fromfile(
                self.indices_path, dtype=np.uint32, count=self.n_samples
            )
            self.features = np.memmap(
                self.global_features_path,
                dtype=self.features_dtype,
                mode="r",
                shape=(self.global_n_samples,),
            )
            self.labels = np.memmap(
                self.global_labels_path,
                dtype=np.float32,
                mode="r",
                shape=(self.global_n_samples, 3),
            )
        else:
            self.features = np.fromfile(
                self.legacy_features_path, dtype=self.features_dtype, count=self.n_samples
            )
            self.labels = np.fromfile(
                self.legacy_labels_path, dtype=np.float32, count=self.n_samples * 3
            ).reshape(-1, 3)
        print("Done.")

    def _lazy_load(self):
        if self.mode == "indices":
            if self.indices is None:
                self.indices = np.memmap(
                    self.indices_path,
                    dtype=np.uint32,
                    mode="r",
                    shape=(self.n_samples,),
                )
            if self.features is None:
                self.features = np.memmap(
                    self.global_features_path,
                    dtype=self.features_dtype,
                    mode="r",
                    shape=(self.global_n_samples,),
                )
            if self.labels is None:
                self.labels = np.memmap(
                    self.global_labels_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self.global_n_samples, 3),
                )
        else:
            if self.features is None:
                self.features = np.memmap(
                    self.legacy_features_path,
                    dtype=self.features_dtype,
                    mode="r",
                    shape=(self.n_samples,),
                )
                self.labels = np.memmap(
                    self.legacy_labels_path, dtype=np.float32, mode="r", shape=(self.n_samples, 3)
                )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._lazy_load()

        if self.mode == "indices":
            global_idx = int(self.indices[idx])
            row = self.features[global_idx]
            label = np.array(self.labels[global_idx], copy=True)
        else:
            row = self.features[idx]
            label = np.array(self.labels[idx], copy=True)

        branches, bypass = decode_feature_row(
            row, self.feature_layout, self.PLANES_PER_TYPE, self.PACKED_OFFSETS,
            self.PACKED_BRANCH_PLANES, self.NUM_BYPASS_PLANES
        )
        global_feats = np.array(row["global"][: self.n_globals], copy=True)
        wdl = label[:3]

        return {
            "branches": torch.from_numpy(branches),
            "bypass": torch.from_numpy(bypass),
            "global": torch.from_numpy(global_feats),
            "wdl": torch.from_numpy(wdl),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--max", type=int, default=1000, help="Max samples")
    parser.add_argument("--n_globals", type=int, default=21)
    args = parser.parse_args()

    print("--- ChessMoEFactorizedDataset Check ---")
    ds = ChessMoEFactorizedDataset(args.data_dir, max_samples=args.max, n_globals=args.n_globals)
    print(f"Dataset length: {len(ds)}")

    sample = ds[0]
    print("\nSample contents:")
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    print(f"\nWDL sum: {sample['wdl'].sum().item():.4f} (should be ~1.0)")
    print(f"Base weights sum: {sample['base_weights'].sum().item():.4f} (should be ~1.0)")
    print(f"Aux weights: {sample['aux_weights'].tolist()} (expected 0..1)")
    print("\nStatus: OK")
