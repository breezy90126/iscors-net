import torch
import numpy as np
from torch.utils.data import Dataset
import random
from utils.traditional_iscors import fit_physical_parameters
from tqdm import tqdm

class TauSparseDataset(Dataset):
    """
    A standalone dataset for Internal Learning on a single video.
    It performs Physics-Informed Temporal Sampling (tau slicing) on the fly.
    """
    def __init__(self, video_tensor, tau_delays=(0, 1, 2, 4, 8, 16, 32, 64), patch_size=64, mode='train', train_ratio=0.01):
        """
        Args:
            video_tensor (np.ndarray or torch.Tensor): The full video matrix of shape (T, H, W).
            tau_delays (tuple): The specific time delays to sample. 
                                Default provides 8 channels capturing exponential time scales.
            patch_size (int): Spatial size of the cropped patches.
            mode (str): 'train' for the 1% internal learning, 'inference' for full frame.
            train_ratio (float): Percentage of pixels to use for training (default 1%).
        """
        super().__init__()
        self.video = video_tensor
        self.T, self.H, self.W = video_tensor.shape
        self.tau_delays = tau_delays
        self.max_tau = max(tau_delays)
        self.patch_size = patch_size
        self.mode = mode
        
        # We need a margin so the spatial crop doesn't go out of bounds
        self.margin = patch_size // 2
        
        # Calculate valid spatial and temporal ranges
        self.valid_y = range(self.margin, self.H - self.margin)
        self.valid_x = range(self.margin, self.W - self.margin)
        self.valid_t = range(0, self.T - self.max_tau)
        
        # Generate the coordinate pool
        # For training, we only pick 1% of the spatial locations
        all_coords = [(y, x) for y in self.valid_y for x in self.valid_x]
        num_train_pixels = int(len(all_coords) * train_ratio)
        
        # Fix seed so the 1% is consistent across epochs
        rng = random.Random(42)
        self.train_coords = rng.sample(all_coords, max(1, num_train_pixels))
        
        # Calculate actual GT using traditional algorithm for the 1% sparse coordinates
        self.gt_gamma = {}
        self.gt_alpha = {}
        
        if self.mode == 'train':
            print(f"Pre-calculating Ground Truth for {len(self.train_coords)} sparse pixels (1%)...")
            for coord in tqdm(self.train_coords):
                y, x = coord
                # Extract the 1D time trace for this specific pixel
                trace = self.video[:, y, x]
                # Run the traditional auto-correlation and curve fitting
                gamma_val, alpha_val = fit_physical_parameters(trace, max_tau=self.max_tau)
                self.gt_gamma[coord] = gamma_val
                self.gt_alpha[coord] = alpha_val
            print("GT pre-calculation complete!")

    def __len__(self):
        if self.mode == 'train':
            # For quick testing, just loop through the coords once per epoch
            return len(self.train_coords)
        else:
            return 1 # Inference usually takes the whole video at once or batches it sequentially

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 1. Randomly pick a spatial coordinate from our 1% pool
            coord_idx = idx % len(self.train_coords)
            y, x = self.train_coords[coord_idx]
            
            # 2. Randomly pick a base time t
            t = random.choice(self.valid_t)
            
            # 3. Physics-Informed Temporal Sampling (tau slicing)
            # We extract exactly the frames defined by tau_delays
            frames = []
            for tau in self.tau_delays:
                frame = self.video[t + tau]
                # Spatial crop
                patch = frame[y - self.margin : y + self.margin, x - self.margin : x + self.margin]
                frames.append(patch)
            
            # Stack into (Channels, H, W) where Channels = len(tau_delays)
            # Input shape will be exactly (8, patch_size, patch_size)
            sparse_tensor = np.stack(frames, axis=0)
            
            # Fetch the pre-calculated actual GT for this specific pixel
            gamma_gt = self.gt_gamma[(y, x)]
            alpha_gt = self.gt_alpha[(y, x)]
            
            # Note: We return the GT as a spatial map (patch_size x patch_size) 
            # In a real scenario, the entire patch might have varying GT, but for internal learning
            # on sparse points, we often just supervise the center pixel, or we precalculate the patch.
            # Here we fill the patch with the center value for simplicity in this MVP stub.
            gamma_map = np.full((self.patch_size, self.patch_size), gamma_gt, dtype=np.float32)
            alpha_map = np.full((self.patch_size, self.patch_size), alpha_gt, dtype=np.float32)
            gt_maps = np.stack([gamma_map, alpha_map], axis=0)
            
            return torch.from_numpy(sparse_tensor).float(), torch.from_numpy(gt_maps).float()
            
        else:
            # Inference mode: return the full spatial frame but still compressed in time
            t = 0 # e.g., starting at t=0
            frames = []
            for tau in self.tau_delays:
                frames.append(self.video[t + tau])
            sparse_tensor = np.stack(frames, axis=0) # (8, H, W)
            return torch.from_numpy(sparse_tensor).float()

if __name__ == "__main__":
    # Test the dataset structure
    dummy_video = np.random.rand(200, 256, 256).astype(np.float32)
    dataset = TauSparseDataset(dummy_video, mode='train')
    print(f"Dataset length (1% subset epochs): {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape (Tau Channels, H, W): {x.shape}")
    print(f"GT shape (Gamma/Alpha, H, W): {y.shape}")
