"""
Dataset for Residual MoE training.

Loads:
  - features.bin: ChessInput (32 planes + 16 globals)
  - labels.bin: WDLOutput (4 floats: win, draw, loss, mate_dist)
  - expert_weights.bin: 6 floats [base0-3, survivor, killer]
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import mmap
import sys


class ChessMoEDataset(Dataset):
    """
    Dataset for Residual MoE training.
    
    Returns dict with:
        - layers: [32, 8, 8] uint8 board planes
        - global: [16] float32 global features
        - wdl: [3] float32 WDL probabilities
        - mate: [1] float32 mate distance
        - base_weights: [4] float32 base expert weights (softmax normalized)
        - aux_weights: [2] float32 aux expert gates [survivor, killer]
    """
    
    def __init__(self, data_dir, max_samples=None, in_memory=False):
        """
        Args:
            data_dir: Directory containing features.bin, labels.bin, expert_weights.bin
            max_samples: Optional limit on number of samples
            in_memory: If True, load everything to RAM
        """
        self.in_memory = in_memory
        self.data_dir = data_dir
        
        # File paths
        self.features_path = os.path.join(data_dir, "features.bin")
        self.labels_path = os.path.join(data_dir, "labels.bin")
        self.weights_path = os.path.join(data_dir, "expert_weights.bin")
        
        # Check files exist
        for path in [self.features_path, self.labels_path, self.weights_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")
        
        # Define C++ struct layout for features
        # Matches: struct alignas(64) ChessInput { uint8_t layers[32][64]; float global[16]; }
        self.features_dtype = np.dtype([
            ('layers', np.uint8, (32, 64)),  # 32 planes * 64 squares = 2048 bytes
            ('global', np.float32, 16),      # 16 globals * 4 bytes = 64 bytes
        ], align=True)
        
        # Calculate sample counts
        feature_size = self.features_dtype.itemsize
        label_size = 4 * 4  # 4 floats (WDL + mate)
        weight_size = 6 * 4  # 6 floats
        
        feature_samples = os.path.getsize(self.features_path) // feature_size
        label_samples = os.path.getsize(self.labels_path) // label_size
        weight_samples = os.path.getsize(self.weights_path) // weight_size
        
        # Verify alignment
        assert feature_samples == label_samples == weight_samples, \
            f"Sample count mismatch: features={feature_samples}, labels={label_samples}, weights={weight_samples}"
        
        self.n_samples = feature_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
            print(f"Dataset: Truncated to {self.n_samples} samples")
        else:
            print(f"Dataset: {self.n_samples:,} samples")
        
        # Lazy-loaded memmaps (for multiprocessing compatibility)
        self.features = None
        self.labels = None
        self.weights = None
        
        if in_memory:
            self._load_to_memory()
    
    def _load_to_memory(self):
        """Load all data to RAM."""
        print("Loading dataset to RAM...")
        self.features = np.fromfile(self.features_path, dtype=self.features_dtype, count=self.n_samples)
        self.labels = np.fromfile(self.labels_path, dtype=np.float32, count=self.n_samples * 4).reshape(-1, 4)
        self.weights = np.fromfile(self.weights_path, dtype=np.float32, count=self.n_samples * 6).reshape(-1, 6)
        print("Done.")
    
    def _lazy_load(self):
        """Lazy-load memmaps (called per worker in multiprocessing)."""
        if self.features is None:
            self.features = np.memmap(self.features_path, dtype=self.features_dtype, mode='r', shape=(self.n_samples,))
            self.labels = np.memmap(self.labels_path, dtype=np.float32, mode='r', shape=(self.n_samples, 4))
            self.weights = np.memmap(self.weights_path, dtype=np.float32, mode='r', shape=(self.n_samples, 6))
            
            # Optimization: hint OS for random access
            for arr in [self.features, self.labels, self.weights]:
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
        layers = np.array(row['layers'])  # Copy for memmap
        global_feats = np.array(row['global'][:15])  # Only 15 valid globals (index 15 is unused padding)
        
        # Labels (WDL + mate)
        label = np.array(self.labels[idx])
        wdl = label[:3]  # [win, draw, loss]
        mate = label[3:4]  # [mate_dist]
        
        # Expert weights
        weights = np.array(self.weights[idx])
        base_weights = weights[:4]  # [tactical, strategic, major, minor]
        aux_weights = weights[4:6]  # [survivor, killer]
        
        return {
            'layers': torch.from_numpy(layers).view(32, 8, 8),  # [32, 8, 8]
            'global': torch.from_numpy(global_feats),           # [16]
            'wdl': torch.from_numpy(wdl),                       # [3]
            'mate': torch.from_numpy(mate),                     # [1]
            'base_weights': torch.from_numpy(base_weights),     # [4]
            'aux_weights': torch.from_numpy(aux_weights),       # [2]
        }
    
    def get_expert_mask(self, expert_idx, threshold=0.5):
        """
        Get indices where a specific expert is primary.
        
        Args:
            expert_idx: 0-3 for base experts, 4-5 for aux
            threshold: For aux experts, activation threshold
        """
        self._lazy_load()
        
        if expert_idx < 4:
            # Base expert: check argmax
            base = self.weights[:, :4]
            primary = np.argmax(base, axis=1)
            return np.where(primary == expert_idx)[0]
        else:
            # Aux expert: check threshold
            aux_idx = expert_idx - 4
            aux = self.weights[:, 4 + aux_idx]
            return np.where(aux > threshold)[0]


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--max", type=int, default=1000, help="Max samples")
    args = parser.parse_args()
    
    print("--- ChessMoEDataset Check ---")
    
    ds = ChessMoEDataset(args.data_dir, max_samples=args.max)
    print(f"Dataset length: {len(ds)}")
    
    # Get one sample
    sample = ds[0]
    print("\nSample contents:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    
    # Check WDL sums to 1
    print(f"\nWDL sum: {sample['wdl'].sum().item():.4f} (should be 1.0)")
    
    # Check base weights sum to 1
    print(f"Base weights sum: {sample['base_weights'].sum().item():.4f} (should be 1.0)")
    
    # Check aux weights range
    print(f"Aux weights: {sample['aux_weights'].tolist()} (should be 0-1)")
    
    print("\nStatus: OK ✓")


# =============================================================================
#   EXPERT-SPECIFIC DATASET (for Stage 2 training)
# =============================================================================
class ChessExpertDataset(Dataset):
    """
    Dataset for training a single expert on pre-split data.
    
    Loads expert-specific files created by C++ preprocessing:
      - expert{N}_features.bin
      - expert{N}_labels.bin
    
    No filtering needed - data is already routed to appropriate expert.
    
    Usage:
        ds = ChessExpertDataset("../data/processed", expert_idx=0)  # Tactical
        ds = ChessExpertDataset("../data/processed", expert_idx=4)  # Survivor
    """
    
    EXPERT_NAMES = ['Tactical', 'Strategic', 'Major End', 'Minor End', 'Survivor', 'Killer']
    
    def __init__(self, data_dir, expert_idx, max_samples=None, in_memory=False):
        """
        Args:
            data_dir: Directory containing expertN_features.bin, expertN_labels.bin
            expert_idx: 0-5 for which expert's dataset to load
            max_samples: Optional limit on number of samples
            in_memory: If True, load everything to RAM
        """
        self.in_memory = in_memory
        self.data_dir = data_dir
        self.expert_idx = expert_idx
        self.expert_name = self.EXPERT_NAMES[expert_idx]
        
        # File paths for this expert
        self.features_path = os.path.join(data_dir, f"expert{expert_idx}_features.bin")
        self.labels_path = os.path.join(data_dir, f"expert{expert_idx}_labels.bin")
        
        # Check files exist
        for path in [self.features_path, self.labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing expert file: {path}")
        
        # Define C++ struct layout for features (same as main dataset)
        self.features_dtype = np.dtype([
            ('layers', np.uint8, (32, 64)),  # 32 planes * 64 squares
            ('global', np.float32, 16),      # 16 globals
        ], align=True)
        
        # Calculate sample counts
        feature_size = self.features_dtype.itemsize
        label_size = 4 * 4  # 4 floats (WDL + mate)
        
        feature_samples = os.path.getsize(self.features_path) // feature_size
        label_samples = os.path.getsize(self.labels_path) // label_size
        
        assert feature_samples == label_samples, \
            f"Sample count mismatch: features={feature_samples}, labels={label_samples}"
        
        self.n_samples = feature_samples
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
        
        print(f"Expert Dataset [{expert_idx}] {self.expert_name}: {self.n_samples:,} samples")
        
        # Lazy-loaded memmaps
        self.features = None
        self.labels = None
        
        if in_memory:
            self._load_to_memory()
    
    def _load_to_memory(self):
        """Load all data to RAM."""
        print(f"Loading expert {self.expert_idx} to RAM...")
        self.features = np.fromfile(self.features_path, dtype=self.features_dtype, count=self.n_samples)
        self.labels = np.fromfile(self.labels_path, dtype=np.float32, count=self.n_samples * 4).reshape(-1, 4)
        print("Done.")
    
    def _lazy_load(self):
        """Lazy-load memmaps (called per worker in multiprocessing)."""
        if self.features is None:
            self.features = np.memmap(self.features_path, dtype=self.features_dtype, 
                                      mode='r', shape=(self.n_samples,))
            self.labels = np.memmap(self.labels_path, dtype=np.float32, 
                                    mode='r', shape=(self.n_samples, 4))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._lazy_load()
        
        # Features
        row = self.features[idx]
        layers = np.array(row['layers'])
        global_feats = np.array(row['global'][:15])  # Only 15 valid globals
        
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

