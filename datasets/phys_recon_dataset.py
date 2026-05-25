import numpy as np
import torch
import random
from torch.utils.data import Dataset

from utils.traditional_iscors import compute_g_empirical_map


class PhysReconDataset(Dataset):
    """
    Dataset for v3.4 Physics Reconstruction training.

    v3.4 = v3.1 loss (shape-only normalised MSE)
         + v3.3 input mechanism (masked G_empirical as model input)

    Per-pixel normalisation:
        G_norm(τ; y,x) = G_empirical(τ; y,x) / G_empirical(τ₁; y,x)
    Both input and loss target are in this normalised shape space.
    Amplitude A is completely removed from the model — no mean-regression
    attractor, no amplitude-dominated gradient.

    Spatial blind-spot:
        80% of cell pixels: G_norm visible in input → loss applied.
        20% of cell pixels: G_norm zeroed in input → excluded from loss.
    The model must infer held-out pixels from neighbouring visible pixels,
    giving a meaningful spatial generalisation test.

    Returns:
        g_input  : (K, P, P) masked normalised G — model input
        g_target : (P, P, K) full normalised G   — loss target
        train_mask: (P, P)   1 at visible pixels
    """

    def __init__(self,
                 video_tensor,
                 recon_taus=(1, 2, 4, 8, 16, 32, 48, 64),
                 patch_size=64,
                 mode='train',
                 train_fraction=0.80,
                 min_cv=0.005,
                 seed=42,
                 # legacy args — kept so existing call-sites don't break
                 random_tau=True,
                 num_tau_channels=8,
                 max_tau=64):
        super().__init__()
        self.video = video_tensor.astype(np.float32)
        self.T, self.H, self.W = self.video.shape
        self.recon_taus = tuple(int(t) for t in recon_taus)
        self.K = len(self.recon_taus)
        self.patch_size = patch_size
        self.mode = mode
        self.margin = patch_size // 2

        # ---- Precompute G_empirical -----------------------------------------
        print(f"[PhysRecon] Precomputing G_empirical at τ={list(self.recon_taus)} ...")
        g_empirical, self.cell_mask = compute_g_empirical_map(
            self.video, self.recon_taus, min_cv=min_cv
        )
        n_cell = int(self.cell_mask.sum())
        print(f"[PhysRecon] Cell pixels: {n_cell}/{self.cell_mask.size}  "
              f"({100*n_cell/self.cell_mask.size:.1f}%)")

        # ---- Per-pixel normalisation: shape only, amplitude removed ---------
        # G_norm(τ) = G_empirical(τ) / G_empirical(τ₁).
        # Background pixels (G=0) normalise to 0; cell pixels start at 1.
        eps = 1e-10
        g_tau1 = g_empirical[:, :, 0:1]                     # (H, W, 1)
        self.g_norm = g_empirical / (g_tau1 + eps)           # (H, W, K)
        self.g_norm[~self.cell_mask] = 0.0                   # zero background

        # ---- 80 / 20 split on CELL pixels -----------------------------------
        rng = random.Random(seed)
        cell_coords = list(zip(*np.where(self.cell_mask)))
        rng.shuffle(cell_coords)
        n_train = int(len(cell_coords) * train_fraction)
        train_set = set(cell_coords[:n_train])
        eval_set  = set(cell_coords[n_train:])

        self.supervised_mask = np.zeros((self.H, self.W), dtype=np.float32)
        for (y, x) in train_set:
            self.supervised_mask[y, x] = 1.0

        self.held_out_mask = np.zeros((self.H, self.W), dtype=bool)
        for (y, x) in eval_set:
            self.held_out_mask[y, x] = True

        # Masked normalised G used as model input: held-out pixels zeroed
        self.g_norm_masked = self.g_norm.copy()
        self.g_norm_masked[self.held_out_mask] = 0.0

        if mode == 'train':
            valid = [(y, x) for (y, x) in train_set
                     if self.margin <= y < self.H - self.margin
                     and self.margin <= x < self.W - self.margin]
            self.coords = valid
            print(f"[PhysRecon] Train: {len(valid)} supervised patches  "
                  f"(held-out cell pixels: {len(eval_set)})")
        else:
            self.coords = []

    def __len__(self):
        return len(self.coords) if self.mode == 'train' else 1

    def __getitem__(self, idx):
        if self.mode != 'train':
            return self._get_full_frame()

        y, x = self.coords[idx % len(self.coords)]
        m = self.margin

        # Input: masked normalised G patch → (K, P, P)
        g_in  = self.g_norm_masked[y-m:y+m, x-m:x+m]        # (P, P, K)
        g_input = torch.from_numpy(
            g_in.transpose(2, 0, 1).astype(np.float32))       # (K, P, P)

        # Target: full normalised G patch → (P, P, K)
        g_tgt  = self.g_norm[y-m:y+m, x-m:x+m]
        g_target = torch.from_numpy(g_tgt.astype(np.float32))

        # Supervision mask: 1 at visible (80%) pixels
        mask   = self.supervised_mask[y-m:y+m, x-m:x+m]
        mask_t = torch.from_numpy(mask)

        return g_input, g_target, mask_t

    def _get_full_frame(self):
        # Inference: full masked normalised G map → (K, H, W)
        g = self.g_norm_masked.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(g)


if __name__ == "__main__":
    dummy = np.random.randn(200, 128, 128).astype(np.float32) + 100.0
    ds = PhysReconDataset(dummy, recon_taus=(1, 2, 4, 8, 16, 32, 48, 64))
    print(f"Dataset length: {len(ds)}")
    g_in, g_tgt, mask = ds[0]
    print(f"Input  (K,P,P)   : {g_in.shape}  range [{g_in.min():.3f}, {g_in.max():.3f}]")
    print(f"Target (P,P,K)   : {g_tgt.shape}  range [{g_tgt.min():.3f}, {g_tgt.max():.3f}]")
    print(f"Mask   (P,P)     : {mask.shape}   visible fraction: {mask.mean():.3f}")
