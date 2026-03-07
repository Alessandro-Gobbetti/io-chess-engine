"""
Simplified dataset for WDL training.

Loads from train/ or val/ directory with separate .bin files.
No expert weights needed (single head model).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import mmap


class ChessSimpleDataset(Dataset):
    """
    Simplified dataset for single-head WDL training.
    
    Loads from either train/ or val/ directory:
        - features.bin: ChessInput (32 planes + 16 globals)
        - labels.bin: WDLOutput (4 floats: win, draw, loss, mate_dist)
    """
    
    def __init__(self, data_dir, split='train', max_samples=None, in_memory=False):
        """
        Args:
            data_dir: Base directory containing train/ and val/ subdirs
            split: 'train' or 'val'
            max_samples: Optional limit on samples
            in_memory: If True, load to RAM
        """
        self.in_memory = in_memory
        self.split_dir = os.path.join(data_dir, split)
        
        # File paths
        self.features_path = os.path.join(self.split_dir, "features.bin")
        self.labels_path = os.path.join(self.split_dir, "labels.bin")
        
        # Check files exist
        for path in [self.features_path, self.labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
        
        # C++ struct layout (matches ChessInput)
        self.features_dtype = np.dtype([
            ('layers', np.uint8, (32, 64)),  # 32 planes * 64 squares
            ('global', np.float32, 16),      # 16 globals (only 15 used)
        ], align=True)
        
        # Calculate sample counts
        feature_samples = os.path.getsize(self.features_path) // self.features_dtype.itemsize
        label_samples = os.path.getsize(self.labels_path) // (4 * 4)  # 4 floats
        
        assert feature_samples == label_samples, \
            f"Sample mismatch: features={feature_samples}, labels={label_samples}"
        
        self.n_samples = feature_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
            
        print(f"Dataset [{split}]: {self.n_samples:,} samples from {self.split_dir}")
        
        # Lazy-loaded memmaps
        self.features = None
        self.labels = None
        
        if in_memory:
            self._load_to_memory()
    
    def _load_to_memory(self):
        """Load all data to RAM."""
        print("Loading to RAM...")
        self.features = np.fromfile(self.features_path, dtype=self.features_dtype, count=self.n_samples)
        self.labels = np.fromfile(self.labels_path, dtype=np.float32, count=self.n_samples * 4).reshape(-1, 4)
        print("Done.")
    
    def _lazy_load(self):
        """Lazy-load memmaps."""
        if self.features is None:
            self.features = np.memmap(self.features_path, dtype=self.features_dtype, mode='r', shape=(self.n_samples,))
            self.labels = np.memmap(self.labels_path, dtype=np.float32, mode='r', shape=(self.n_samples, 4))
            
            # Hint OS for random access
            for arr in [self.features, self.labels]:
                if hasattr(arr, '_mmap') and arr._mmap:
                    try:
                        arr._mmap.madvise(mmap.MADV_RANDOM)
                    except:
                        pass
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._lazy_load()
        
        # Features
        row = self.features[idx]
        layers = np.array(row['layers'])
        global_feats = np.array(row['global'][:15])  # 15 valid globals
        
        # Labels (WDL + mate)
        label = np.array(self.labels[idx])
        wdl = label[:3]
        mate = label[3:4]
        
        return {
            'layers': torch.from_numpy(layers).view(32, 8, 8),
            'global': torch.from_numpy(global_feats),
            'wdl': torch.from_numpy(wdl),
            'mate': torch.from_numpy(mate),
        }


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max", type=int, default=1000)
    args = parser.parse_args()
    
    print("--- ChessSimpleDataset Check ---")
    
    ds = ChessSimpleDataset(args.data_dir, split=args.split, max_samples=args.max)
    print(f"Dataset length: {len(ds)}")
    
    # Get one sample
    sample = ds[0]
    print("\nSample contents:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    
    print(f"\nWDL: {sample['wdl'].numpy()}")
    print(f"WDL sum: {sample['wdl'].sum().item():.4f}")
    print(f"Mate: {sample['mate'].item():.4f}")
