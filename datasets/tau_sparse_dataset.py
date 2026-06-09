import torch
import numpy as np
from torch.utils.data import Dataset
import random
from utils.traditional_iscors import fit_physical_parameters
from tqdm import tqdm

class TauSparseDataset(Dataset):
    """
    Dataset for Internal Learning on a single video.
    Physics-Informed Temporal Sampling with optional random tau selection
    and center 3x3 supervised mask for sparse pixel supervision.

    When gt_npz_path is provided (synthetic validation), GT is loaded directly
    from the perfect simulation labels — no curve fitting needed.
    When gt_npz_path is None (real data), GT is estimated per-pixel via curve fitting.
    """
    def __init__(self, video_tensor, tau_delays=(0, 1, 2, 4, 8, 16, 32, 64),
                 patch_size=64, mode='train', train_ratio=0.01,
                 random_tau=True, num_tau_channels=8, max_tau=64,
                 gt_npz_path=None, r2_threshold=0.7):
        """
        Args:
            video_tensor:   (T, H, W) float array.
            tau_delays:     fixed tau schedule for inference (or training if random_tau=False).
            patch_size:     spatial crop size (square).
            mode:           'train' or 'inference'.
            train_ratio:    fraction of spatial pixels used for supervision.
            random_tau:     if True, randomly sample num_tau_channels taus per sample.
            num_tau_channels: number of tau channels fed to the model.
            max_tau:        maximum possible tau delay.
            gt_npz_path:    path to _gt.npz for synthetic validation (perfect GT, no noise).
                            Set to None for real data → GT estimated via curve fitting.
            r2_threshold:   (real data only) minimum R² of the autocorrelation fit to accept
                            a pixel as a training supervision point.  Pixels below this are
                            excluded — the model fills them in via spatial generalisation.
                            Ignored when gt_npz_path is provided.
        """
        super().__init__()
        self.video = video_tensor
        self.T, self.H, self.W = video_tensor.shape
        self.tau_delays = tau_delays
        self.max_tau = max_tau
        self.patch_size = patch_size
        self.mode = mode
        self.random_tau = random_tau
        self.num_tau_channels = num_tau_channels

        self.margin = patch_size // 2
        self.valid_y = range(self.margin, self.H - self.margin)
        self.valid_x = range(self.margin, self.W - self.margin)
        # Ensure t + max_tau < T
        self.valid_t = list(range(0, self.T - self.max_tau))

        all_coords = [(y, x) for y in self.valid_y for x in self.valid_x]
        num_train_pixels = int(len(all_coords) * train_ratio)

        # Fixed seed so the sparse subset is consistent across epochs
        rng = random.Random(42)
        self.train_coords = rng.sample(all_coords, max(1, num_train_pixels))

        self.gt_gamma = {}
        self.gt_alpha = {}

        if self.mode == 'train':
            if gt_npz_path is not None:
                # Perfect GT from simulation — zero noise, no curve fitting needed
                print(f"Loading perfect GT from {gt_npz_path} ...")
                gt = np.load(gt_npz_path)
                gt_gamma_map = gt["gamma"]
                gt_alpha_map = gt["alpha"]
                for coord in self.train_coords:
                    y, x = coord
                    self.gt_gamma[coord] = float(gt_gamma_map[y, x])
                    self.gt_alpha[coord] = float(gt_alpha_map[y, x])
                print(f"Perfect GT loaded for {len(self.train_coords)} pixels.")
            else:
                # Real data: estimate GT per-pixel via curve fitting + R² quality filter
                print(f"Pre-calculating GT for {len(self.train_coords)} sparse pixels "
                      f"({train_ratio*100:.1f}%, R²≥{r2_threshold})...")
                gt_r2 = {}
                for coord in tqdm(self.train_coords):
                    y, x = coord
                    trace = self.video[:, y, x]
                    gamma_val, alpha_val, r2 = fit_physical_parameters(trace, max_tau=self.max_tau)
                    self.gt_gamma[coord] = gamma_val
                    self.gt_alpha[coord] = alpha_val
                    gt_r2[coord]         = r2
                print("GT pre-calculation complete!")

                # Accept pixels that are either:
                #   1. Background: CV filter returned (0, 0) — R² irrelevant.
                #   2. Cell with reliable fit: physical range AND R² ≥ threshold.
                # Pixels with bad R² are excluded — model fills them via spatial learning.
                MAX_GAMMA = 2.0
                valid = []
                for c in self.train_coords:
                    is_bg   = (self.gt_gamma[c] == 0.0 and self.gt_alpha[c] == 0.0)
                    is_cell = (0 < self.gt_gamma[c] <= MAX_GAMMA
                               and 0 < self.gt_alpha[c] <= 2.0
                               and gt_r2[c] >= r2_threshold)
                    if is_bg or is_cell:
                        valid.append(c)

                n_bg      = sum(1 for c in valid if self.gt_gamma[c] == 0.0)
                n_cell    = len(valid) - n_bg
                n_low_r2  = sum(1 for c in self.train_coords
                                if self.gt_gamma[c] != 0.0 and gt_r2[c] < r2_threshold)
                print(f"Valid GT pixels: {len(valid)}/{len(self.train_coords)}  "
                      f"(background={n_bg}, cell={n_cell}, excluded low-R²={n_low_r2})")
                if valid:
                    self.train_coords = valid

            gamma_vals = [self.gt_gamma[c] for c in self.train_coords]
            alpha_vals = [self.gt_alpha[c] for c in self.train_coords]
            import statistics
            print(f"GT Gamma — mean={statistics.mean(gamma_vals):.3f}  "
                  f"min={min(gamma_vals):.3f}  max={max(gamma_vals):.3f}")
            print(f"GT Alpha — mean={statistics.mean(alpha_vals):.3f}  "
                  f"min={min(alpha_vals):.3f}  max={max(alpha_vals):.3f}")

    def _sample_taus(self):
        """Randomly sample num_tau_channels unique tau values from [0, max_tau]."""
        pool = range(0, self.max_tau + 1)
        return sorted(random.sample(pool, min(self.num_tau_channels, len(pool))))

    def _center_mask(self):
        """Mask with 1s only at the center 3x3 pixels; zero elsewhere."""
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        cy, cx = self.patch_size // 2, self.patch_size // 2
        mask[cy - 1:cy + 2, cx - 1:cx + 2] = 1.0
        return mask

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_coords)
        return 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            y, x = self.train_coords[idx % len(self.train_coords)]
            t = random.choice(self.valid_t)

            taus = self._sample_taus() if self.random_tau else list(self.tau_delays)

            frames = []
            for tau in taus:
                frame = self.video[t + tau]
                patch = frame[y - self.margin: y + self.margin,
                              x - self.margin: x + self.margin]
                frames.append(patch)

            # (num_tau_channels, P, P)
            sparse_tensor = np.stack(frames, axis=0).astype(np.float32)

            # Normalize per-sample so the network sees fluctuations, not DC level
            mu = sparse_tensor.mean()
            sig = sparse_tensor.std() + 1e-8
            sparse_tensor = (sparse_tensor - mu) / sig

            gamma_gt = self.gt_gamma[(y, x)]
            alpha_gt = self.gt_alpha[(y, x)]

            # GT maps filled with center pixel value; only center 3x3 is trained via mask
            gamma_map = np.full((self.patch_size, self.patch_size), gamma_gt, dtype=np.float32)
            alpha_map = np.full((self.patch_size, self.patch_size), alpha_gt, dtype=np.float32)
            gt_maps = np.stack([gamma_map, alpha_map], axis=0)  # (2, P, P)

            mask = self._center_mask()  # (P, P)

            return (torch.from_numpy(sparse_tensor),
                    torch.from_numpy(gt_maps),
                    torch.from_numpy(mask))

        else:
            # Inference: full frame using fixed tau_delays
            t = 0
            frames = [self.video[t + tau] for tau in self.tau_delays]
            sparse_tensor = np.stack(frames, axis=0).astype(np.float32)
            # Same normalization as training — critical to avoid distribution mismatch
            mu = sparse_tensor.mean()
            sig = sparse_tensor.std() + 1e-8
            sparse_tensor = (sparse_tensor - mu) / sig
            return torch.from_numpy(sparse_tensor)


if __name__ == "__main__":
    dummy_video = np.random.rand(200, 256, 256).astype(np.float32)
    dataset = TauSparseDataset(dummy_video, mode='train', train_ratio=0.02,
                               random_tau=True, num_tau_channels=8, max_tau=64)
    print(f"Dataset length: {len(dataset)}")
    x, y, m = dataset[0]
    print(f"Input  (Tau, H, W): {x.shape}")
    print(f"GT     (2, H, W):   {y.shape}")
    print(f"Mask   (H, W):      {m.shape}, nonzero={m.sum().item()}")
