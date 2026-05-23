import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from .data_process import load3DImages2Tensor

class ISCATSequenceDataset(Dataset):
    """
    Dataset that returns temporal sequences of image stacks for MSD calculation.
    """
    def __init__(self, dataPath, mode='train', dataExtension='tif', 
                 stack_size=5, sequence_length=10, transform=None):
        """
        dataPath: path to folder containing TIF files
        stack_size: number of frames input to the network (e.g. 5)
        sequence_length: number of consecutive stacks to return (e.g. 10)
                         for computing MSD over tau=1..sequence_length
        """
        self.dataPath = dataPath
        self.mode = mode
        self.stack_size = stack_size
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load all images (Tensor [T, H, W])
        self.image_list = load3DImages2Tensor(dataPath, dataExtension)
        
        # Load trajectories if available
        self.traj_list = []
        tif_files = sorted(glob.glob(os.path.join(dataPath, f"*.{dataExtension}")))
        for tif in tif_files:
            traj_path = tif.replace(f'.{dataExtension}', '_traj.npy')
            if os.path.exists(traj_path):
                self.traj_list.append(np.load(traj_path))
            else:
                self.traj_list.append(None)
        
        self.samples = []
        for img_idx, img in enumerate(self.image_list):
            T_movie, H, W = img.shape
            # We can extract trajectories of length `sequence_length`
            # Each point in trajectory needs `stack_size` frames context.
            # Total frames needed: sequence_length + stack_size - 1 (if reusing context?)
            # No, if stack is centered at t, we need t - window//2 to t + window//2.
            # Let's say we want latent z(t), z(t+1), ... z(t+L-1).
            # We need stacks centered at these times.
            # Max time needed: (t+L-1) + window//2
            # Min time needed: t - window//2
            
            radius = stack_size // 2
            start_t = radius
            end_t = T_movie - radius - sequence_length
            
            if end_t <= start_t:
                continue
                
            # Stride? 
            for t in range(start_t, end_t, 5): # Stride 5 to reduce data
                self.samples.append((img_idx, t))
                
        if not self.samples:
            total_required = sequence_length + stack_size - 1
            raise ValueError(
                f"No samples could be extracted from {len(self.image_list)} movies. "
                f"Possible reason: Movie length is too short for requested sequence_length={sequence_length} "
                f"and stack_size={stack_size}. Required minimum frames: {total_required}."
            )
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_idx, start_t = self.samples[idx]
        img = self.image_list[img_idx]
        
        # Extract sequence of stacks
        # Output: [Sequence_Length, Stack_Size, H, W]
        radius = self.stack_size // 2
        
        stacks = []
        for i in range(self.sequence_length):
            t = start_t + i
            # Stack from t-radius to t+radius+1 (if odd)
            # FAST usually takes odd stack size 
            stack = img[t - radius : t + radius + 1, :, :]
            stacks.append(stack)
            
        # GT Trajectory
        traj = self.traj_list[img_idx]
        if traj is not None:
            # Extract corresponding z-trajectory
            # [T, N, 3] -> [Sequence_Length, N, 3]
            # We focus on Z (index 2)
            gt_z = traj[start_t : start_t + self.sequence_length, :, 2]
            gt_z = torch.from_numpy(gt_z).float()
        else:
            gt_z = torch.zeros(self.sequence_length, 1) # Dummy

        return torch.stack(stacks, dim=0), gt_z
