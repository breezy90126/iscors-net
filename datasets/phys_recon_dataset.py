import numpy as np
import torch
import random
from torch.utils.data import Dataset

from utils.traditional_iscors import compute_g_empirical_map


class PhysReconDataset(Dataset):
    """
    Dataset for v3.0 Physics Reconstruction training.

    Key design (vs TauSparseDataset):
      - GT target is G_empirical(τ) per pixel (precomputed from the same video),
        not (γ, α) labels from curve_fit.  No external labels needed.
      - 80 / 20 split of cell pixels:
          * 80% → supervised pixels (physics reconstruction loss applied)
          * 20% → held-out pixels (never see loss; used for generalisation MAE)
      - Each sample returns: tau-slice input, G_empirical patch (P,P,K),
        and a train_mask (P,P) marking supervised pixels in this patch.

    The mask matters because patches centred on supervised pixels still
    *contain* held-out pixels inside the receptive field — those must be
    excluded from the loss to prevent leakage.
    """

    def __init__(self,
                 video_tensor,
                 recon_taus=(1, 2, 4, 8, 16, 32, 48, 64),
                 patch_size=64,
                 mode='train',
                 train_fraction=0.80,
                 random_tau=True,
                 num_tau_channels=8,
                 max_tau=64,
                 min_cv=0.005,
                 seed=42):
        super().__init__()
        self.video = video_tensor.astype(np.float32)
        self.T, self.H, self.W = self.video.shape
        self.recon_taus = tuple(int(t) for t in recon_taus)
        self.patch_size = patch_size
        self.mode = mode
        self.random_tau = random_tau
        self.num_tau_channels = num_tau_channels
        self.max_tau = max_tau

        self.margin = patch_size // 2
        self.valid_t = list(range(0, self.T - self.max_tau))

        # ---- Precompute G_empirical for every pixel -------------------------
        print(f"[PhysRecon] Precomputing G_empirical at τ={list(self.recon_taus)} ...")
        self.g_empirical, self.cell_mask = compute_g_empirical_map(
            self.video, self.recon_taus, min_cv=min_cv
        )
        n_cell = int(self.cell_mask.sum())
        print(f"[PhysRecon] Cell pixels: {n_cell}/{self.cell_mask.size}  "
              f"({100*n_cell/self.cell_mask.size:.1f}%)")

        # ---- 80 / 20 split on CELL pixels only ------------------------------
        # Background already has G=0 and is handled by post-inference masking.
        rng = random.Random(seed)
        cell_coords = list(zip(*np.where(self.cell_mask)))
        rng.shuffle(cell_coords)
        n_train = int(len(cell_coords) * train_fraction)
        train_set = set(cell_coords[:n_train])
        eval_set  = set(cell_coords[n_train:])

        # supervised_mask: 1 only at the 80% supervised cell pixels
        self.supervised_mask = np.zeros((self.H, self.W), dtype=np.float32)
        for (y, x) in train_set:
            self.supervised_mask[y, x] = 1.0
        # held_out_mask for evaluation
        self.held_out_mask = np.zeros((self.H, self.W), dtype=bool)
        for (y, x) in eval_set:
            self.held_out_mask[y, x] = True

        # Coordinates we will iterate over as patch centres
        if mode == 'train':
            # Only sample patches centred on supervised pixels
            valid = [(y, x) for (y, x) in train_set
                     if self.margin <= y < self.H - self.margin
                     and self.margin <= x < self.W - self.margin]
            self.coords = valid
            print(f"[PhysRecon] Train: {len(valid)} supervised patches  "
                  f"(held-out cell pixels: {len(eval_set)})")
        else:
            # 'inference' mode: a single full-frame sample
            self.coords = []

    def _sample_taus(self):
        pool = range(0, self.max_tau + 1)
        return sorted(random.sample(pool, min(self.num_tau_channels, len(pool))))

    def __len__(self):
        return len(self.coords) if self.mode == 'train' else 1

    def __getitem__(self, idx):
        if self.mode != 'train':
            return self._get_full_frame()

        y, x = self.coords[idx % len(self.coords)]
        t = random.choice(self.valid_t)
        taus = self._sample_taus() if self.random_tau else list(self.recon_taus[:self.num_tau_channels])

        # ---- Input tau-slices --------------------------------------------
        frames = [self.video[t + tau, y - self.margin: y + self.margin,
                             x - self.margin: x + self.margin]
                  for tau in taus]
        sparse_tensor = np.stack(frames, axis=0).astype(np.float32)
        mu = sparse_tensor.mean()
        sig = sparse_tensor.std() + 1e-8
        sparse_tensor = (sparse_tensor - mu) / sig            # (C, P, P)

        # ---- G_empirical patch (target) ----------------------------------
        g_patch = self.g_empirical[y - self.margin: y + self.margin,
                                   x - self.margin: x + self.margin]  # (P, P, K)

        # ---- Train mask (supervised pixels in this patch) ----------------
        mask = self.supervised_mask[y - self.margin: y + self.margin,
                                    x - self.margin: x + self.margin]  # (P, P)

        return (torch.from_numpy(sparse_tensor),
                torch.from_numpy(g_patch.astype(np.float32)),
                torch.from_numpy(mask))

    def _get_full_frame(self):
        # Same fixed-τ inference path as TauSparseDataset, so the model receives
        # an input distribution similar to training time.
        taus = (0, 9, 18, 27, 36, 45, 54, 64)[:self.num_tau_channels]
        frames = [self.video[tau] for tau in taus]
        sparse_tensor = np.stack(frames, axis=0).astype(np.float32)
        mu = sparse_tensor.mean()
        sig = sparse_tensor.std() + 1e-8
        sparse_tensor = (sparse_tensor - mu) / sig
        return torch.from_numpy(sparse_tensor)


if __name__ == "__main__":
    dummy = np.random.randn(200, 128, 128).astype(np.float32) + 100.0
    ds = PhysReconDataset(dummy, max_tau=32)
    print(f"Dataset length: {len(ds)}")
    x, g, m = ds[0]
    print(f"Input shape: {x.shape}")
    print(f"G_empirical patch shape: {g.shape}")
    print(f"Train mask shape: {m.shape}  supervised fraction: {m.mean():.3f}")
