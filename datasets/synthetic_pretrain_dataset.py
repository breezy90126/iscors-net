import numpy as np
import torch
from torch.utils.data import Dataset
import random
from utils.generate_test_video import generate_signal_with_autocorr


class SyntheticPretrainDataset(Dataset):
    """
    Generates on-the-fly synthetic tau-slice patches with known (gamma, alpha).

    Each sample is a patch where every pixel shares the same underlying dynamics
    (plus independent pixel noise), giving DENSE supervision — no sparse mask needed.
    This teaches the model the autocorrelation pattern → (gamma, alpha) mapping
    before it ever sees real data.

    Usage: pretrain once with prior knowledge of the cell's D/alpha range,
    then fine-tune with internal learning on the real video.
    """

    def __init__(self, n_samples=10000,
                 gamma_range=(0.01, 1.0),
                 alpha_range=(0.3, 1.8),
                 T=500,
                 num_tau_channels=8,
                 max_tau=64,
                 patch_size=64,
                 mean_val=100.0,
                 signal_variance=10.0,
                 pixel_noise_std=0.3):
        """
        Args:
            n_samples:         synthetic samples per epoch.
            gamma_range:       (min, max) from prior knowledge of the cell type.
            alpha_range:       (min, max) from prior knowledge.
            T:                 length of synthetic traces.
            pixel_noise_std:   per-pixel spatial noise (relative to signal std).
        """
        self.n_samples = n_samples
        self.gamma_range = gamma_range
        self.alpha_range = alpha_range
        self.T = T
        self.num_tau_channels = num_tau_channels
        self.max_tau = max_tau
        self.patch_size = patch_size
        self.mean_val = mean_val
        self.signal_variance = signal_variance
        self.pixel_noise_std = pixel_noise_std
        self.valid_t = list(range(0, T - max_tau))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        P = self.patch_size

        # Sample physical parameters uniformly from known prior range
        gamma = np.random.uniform(*self.gamma_range)
        alpha = np.random.uniform(*self.alpha_range)

        # Generate a base temporal trace with these dynamics
        base_trace = generate_signal_with_autocorr(
            self.T, gamma, alpha,
            mean_val=self.mean_val,
            variance=self.signal_variance
        )

        # Random time origin and random tau delays (same augmentation as training)
        t = random.choice(self.valid_t)
        taus = sorted(random.sample(range(0, self.max_tau + 1), self.num_tau_channels))

        # Build spatial patch: each pixel = base_trace value + independent pixel noise
        frames = []
        sig_std = np.sqrt(self.signal_variance)
        for tau in taus:
            frame_val = base_trace[t + tau]
            noise = np.random.normal(0.0, self.pixel_noise_std * sig_std, (P, P))
            frame = np.full((P, P), frame_val, dtype=np.float32) + noise.astype(np.float32)
            frames.append(frame)

        sparse_tensor = np.stack(frames, axis=0)  # (C, P, P)

        # Same normalisation as internal learning
        mu = sparse_tensor.mean()
        sig = sparse_tensor.std() + 1e-8
        sparse_tensor = (sparse_tensor - mu) / sig

        # Dense GT (full patch, uniform value) — pretraining has perfect labels
        gamma_map = np.full((P, P), gamma, dtype=np.float32)
        alpha_map = np.full((P, P), alpha, dtype=np.float32)
        gt_maps = np.stack([gamma_map, alpha_map], axis=0)

        # Full mask: supervise every pixel during pretraining
        mask = np.ones((P, P), dtype=np.float32)

        return (torch.from_numpy(sparse_tensor),
                torch.from_numpy(gt_maps),
                torch.from_numpy(mask))
