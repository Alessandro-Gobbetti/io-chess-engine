import numpy as np
import torch
from torch.utils.data import Dataset
import os

class ChessDataset(Dataset):
    def __init__(self, features_path, labels_path, in_memory=False):
        """
        Args:
            in_memory (bool): If True, loads everything into RAM. 
                              If False, reads from disk (memmap).
        """
        self.in_memory = in_memory
        
        # Calculate number of samples based on file size
        # Features: 42 channels * 8 * 8 * 1 byte (uint8) = 2688 bytes per sample
        feature_bytes_per_sample = 42 * 8 * 8
        file_size = os.path.getsize(features_path)
        self.n_samples = file_size // feature_bytes_per_sample

        if in_memory:
            print(f"Loading {features_path} into RAM...")
            # Use fromfile for raw binary data
            self.features = np.fromfile(features_path, dtype=np.uint8)
        else:
            print(f"Mapping {features_path} from Disk...")
            # 'r' mode opens existing file in read-only memmap mode
            self.features = np.memmap(features_path, dtype=np.uint8, mode='r', shape=(self.n_samples, 42, 8, 8))

        # If loaded in memory, reshape now
        if in_memory:
            self.features = self.features.reshape((self.n_samples, 42, 8, 8))

        # labels are always loaded into memory (they are small)
        # Use fromfile for raw binary data
        self.labels = np.fromfile(labels_path, dtype=np.float32)

        # sanity check
        assert self.features.shape[0] == self.labels.shape[0], \
            f"Number of samples in features ({self.features.shape[0]}) and labels ({self.labels.shape[0]}) do not match!"
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the numpy arrays
        f = self.features[idx]
        l = self.labels[idx]

        # If reading from disk (memmap), we MUST copy to detach from the file
        if not self.in_memory:
            f = np.array(f) # Ensure it's a real array, not a memmap view
 
        # Convert to tensor
        return (
            torch.tensor(f, dtype=torch.uint8),  # Convert uint8 to float tensor and normalize
            torch.tensor(l, dtype=torch.float32) # Use torch.tensor for scalar values
        )