import numpy as np
import torch
import random
from torch.utils.data import Dataset

from utils.traditional_iscors import compute_g_empirical_map


class PhysReconDataset(Dataset):
    """
    Dataset for v3.3 Physics Reconstruction training.

    v3.3 key change — model input is masked G_empirical, not raw tau-slices.

    Spatial blind-spot design (analogous to FAST on G maps):
      - G_empirical(τ; y,x) is precomputed once from the full video.
      - 80% of cell pixels are "visible": their G curves appear in the
        model input and contribute to the physics reconstruction loss.
      - 20% of cell pixels are "held-out": their G values are ZEROED in
        the model input and excluded from the loss.
      - The U-Net must infer (γ,α,A) at held-out pixels purely from the
        spatial context of neighbouring visible pixels — spatial redundancy.
      - Validation: compare G_theory predicted at held-out pixels against
        the true G_empirical that was hidden from the input.

    This restores a meaningful 80/20 generalization test:
      v3.0–v3.2 held out pixels from the LOSS but the model's raw-frame
      input still contained them, making the test leaky.
      Here the held-out G is literally absent from the input.

    Return signature: (g_input, g_target, train_mask)
      g_input  : (K, P, P) masked G_empirical — model input
      g_target : (P, P, K) full G_empirical  — loss target
      train_mask: (P, P)   1 at visible pixels, 0 at held-out
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

        # ---- Precompute G_empirical for every pixel -------------------------
        print(f"[PhysRecon] Precomputing G_empirical at τ={list(self.recon_taus)} ...")
        self.g_empirical, self.cell_mask = compute_g_empirical_map(
            self.video, self.recon_taus, min_cv=min_cv
        )
        n_cell = int(self.cell_mask.sum())
        print(f"[PhysRecon] Cell pixels: {n_cell}/{self.cell_mask.size}  "
              f"({100*n_cell/self.cell_mask.size:.1f}%)")

        # ---- 80 / 20 split on CELL pixels only ------------------------------
        rng = random.Random(seed)
        cell_coords = list(zip(*np.where(self.cell_mask)))
        rng.shuffle(cell_coords)
        n_train = int(len(cell_coords) * train_fraction)
        train_set = set(cell_coords[:n_train])
        eval_set  = set(cell_coords[n_train:])

        # supervised_mask: 1.0 at visible (80%) cell pixels
        self.supervised_mask = np.zeros((self.H, self.W), dtype=np.float32)
        for (y, x) in train_set:
            self.supervised_mask[y, x] = 1.0

        # held_out_mask: True at hidden (20%) pixels
        self.held_out_mask = np.zeros((self.H, self.W), dtype=bool)
        for (y, x) in eval_set:
            self.held_out_mask[y, x] = True

        # Masked G_empirical used as model input: held-out G zeroed so the
        # model truly cannot see those pixels' autocorrelation curves.
        self.g_empirical_masked = self.g_empirical.copy()
        self.g_empirical_masked[self.held_out_mask] = 0.0

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

        # Input: masked G_empirical patch → (K, P, P)
        g_in = self.g_empirical_masked[y-m:y+m, x-m:x+m]    # (P, P, K)
        g_input = torch.from_numpy(
            g_in.transpose(2, 0, 1).astype(np.float32))       # (K, P, P)

        # Target: unmasked G_empirical patch → (P, P, K)
        g_patch  = self.g_empirical[y-m:y+m, x-m:x+m]
        g_target = torch.from_numpy(g_patch.astype(np.float32))

        # Supervision mask: 1 at visible pixels, 0 at held-out
        mask = self.supervised_mask[y-m:y+m, x-m:x+m]
        mask_t = torch.from_numpy(mask)

        return g_input, g_target, mask_t

    def _get_full_frame(self):
        # Inference: full masked G_empirical map, shape (K, H, W)
        g = self.g_empirical_masked.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(g)


if __name__ == "__main__":
    dummy = np.random.randn(200, 128, 128).astype(np.float32) + 100.0
    ds = PhysReconDataset(dummy, recon_taus=(1, 2, 4, 8, 16, 32, 48, 64))
    print(f"Dataset length: {len(ds)}")
    g_in, g_tgt, mask = ds[0]
    print(f"Input  (K,P,P)  : {g_in.shape}")
    print(f"Target (P,P,K)  : {g_tgt.shape}")
    print(f"Mask   (P,P)    : {mask.shape}  visible fraction: {mask.mean():.3f}")
