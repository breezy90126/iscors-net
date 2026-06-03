import numpy as np
import torch
import random
from torch.utils.data import Dataset

from utils.traditional_iscors import compute_g_empirical_map
from utils.sn2n_sampling import compute_sigma_g


class PhysReconDataset(Dataset):
    """
    Dataset for v4.3 Physics Reconstruction training.

    v4.3 additions over v4.1:
      - sigma_clip: exclude normalization-unstable pixels.
        Pixels where σ_G_norm(τ_ref) > sigma_clip are removed from cell_mask.
        σ_G_norm(τ_ref) is large when G(τ_ref)≈0 (fast dynamics → blow-up in
        G_norm → loss dominated by noise → γ→0 / α>1 spurious minimum).
        Recommended: sigma_clip=2.0 removes the worst ~20% of unstable pixels.
        Setting sigma_clip=None disables the filter (backward compatible).

    v4.1 additions over v4.0:
      - σ_G(τ; y,x) reliability map computed alongside G_empirical.
        σ²_G(τ) ≈ (2/T) * [G(0)² + G(τ)²] — Wiener-Khinchin noise estimate.
      - σ_G_norm = σ_G / G(τ_ref): noise in the normalised G space (same units
        as G_norm used by the loss).
      - Added as 5th return element so loss can apply Fisher × Reliability
        combined weighting (Fisher: which τ has physics info; Reliability:
        which τ has trustworthy measurements).

    Per-pixel normalisation:
        G_norm(τ; y,x) = G_empirical(τ; y,x) / G_empirical(τ₁; y,x)
    Both input and loss target are in this normalised shape space.

    Spatial blind-spot (v3.9: 35% held-out):
        65% of cell pixels: G_norm visible in input → loss applied.
        35% of cell pixels: G_norm zeroed in input → excluded from loss.

    Returns:
        g_input          : (K, P, P)   masked normalised G — model input
        g_target         : (P, P, K)   full normalised G   — loss target
        train_mask       : (P, P)      1 at supervised (65%) pixels
        cell_mask_patch  : (P, P)      1 at cell pixels (CV ≥ min_cv, σ_clip)
        sigma_g_norm_patch: (P, P, K)  normalised σ_G — reliability map
    """

    def __init__(self,
                 video_tensor,
                 recon_taus=(1, 2, 4, 8, 16, 32, 48, 64),
                 patch_size=64,
                 mode='train',
                 train_fraction=0.65,
                 min_cv=0.005,
                 sigma_clip=None,
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
        eps = 1e-10
        g_tau1 = g_empirical[:, :, 0:1]                     # (H, W, 1)
        self.g_norm = g_empirical / (g_tau1 + eps)           # (H, W, K)
        self.g_norm[~self.cell_mask] = 0.0

        # ---- σ_G reliability map (v4.1) -------------------------------------
        # σ²_G(τ) ≈ (2/T) * [G(0)² + G(τ)²]   — Wiener-Khinchin noise model
        # σ_G_norm(τ) = σ_G(τ) / G(τ_ref)      — noise in normalised G space
        print(f"[PhysRecon] Computing σ_G reliability map ...")
        sigma_g = compute_sigma_g(self.video, self.recon_taus, g_map=g_empirical)
        self.sigma_g_norm = sigma_g / (np.abs(g_tau1) + eps)  # (H, W, K)
        self.sigma_g_norm[~self.cell_mask] = 0.0

        # ---- σ_clip: remove normalization-unstable pixels (v4.3) -----------
        # σ_G_norm(τ_ref) >> 1 means G(τ_ref)≈0 → G_norm blows up → loss noise
        if sigma_clip is not None:
            stable = self.sigma_g_norm[:, :, 0] <= sigma_clip
            n_before = int(self.cell_mask.sum())
            self.cell_mask = self.cell_mask & stable
            n_after  = int(self.cell_mask.sum())
            print(f"[PhysRecon] σ_clip={sigma_clip}: "
                  f"{n_before - n_after} unstable pixels removed "
                  f"→ {n_after}/{self.cell_mask.size} cell pixels "
                  f"({100*n_after/self.cell_mask.size:.1f}%)")
            self.g_norm[~self.cell_mask]       = 0.0
            self.sigma_g_norm[~self.cell_mask] = 0.0

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
        # σ_G_norm is not masked (model does not need it as input by default;
        # it is passed to the loss for reliability-weighted training)
        self.sigma_g_norm_masked = self.sigma_g_norm.copy()
        self.sigma_g_norm_masked[self.held_out_mask] = 0.0

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

        # Supervision mask: 1 at visible (65%) pixels
        mask   = self.supervised_mask[y-m:y+m, x-m:x+m]
        mask_t = torch.from_numpy(mask)

        # Cell mask patch: needed by caller for masked TV (cell-cell pairs only)
        cell_p = self.cell_mask[y-m:y+m, x-m:x+m].astype(np.float32)
        cell_t = torch.from_numpy(cell_p)

        # Reliability: σ_G_norm patch → (P, P, K)
        sg_p = self.sigma_g_norm[y-m:y+m, x-m:x+m].astype(np.float32)
        sg_t = torch.from_numpy(sg_p)

        return g_input, g_target, mask_t, cell_t, sg_t

    def _get_full_frame(self):
        # Inference: full masked normalised G map → (K, H, W)
        g = self.g_norm_masked.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(g)


if __name__ == "__main__":
    dummy = np.random.randn(200, 128, 128).astype(np.float32) + 100.0
    ds = PhysReconDataset(dummy, recon_taus=(1, 2, 4, 8, 16, 32, 48, 64))
    print(f"Dataset length: {len(ds)}")
    g_in, g_tgt, mask, cell, sigma = ds[0]
    print(f"Input   (K,P,P)  : {g_in.shape}  range [{g_in.min():.3f}, {g_in.max():.3f}]")
    print(f"Target  (P,P,K)  : {g_tgt.shape}  range [{g_tgt.min():.3f}, {g_tgt.max():.3f}]")
    print(f"Mask    (P,P)    : {mask.shape}   visible fraction: {mask.mean():.3f}")
    print(f"CellMsk (P,P)    : {cell.shape}   cell fraction: {cell.mean():.3f}")
    print(f"Sigma   (P,P,K)  : {sigma.shape}  range [{sigma.min():.3f}, {sigma.max():.3f}]")
