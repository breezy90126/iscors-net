import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from datasets.tau_sparse_dataset import TauSparseDataset
from models.pissl_tau_encoder import PISSLTauEncoder
from loss.physics_loss import PhysicsInformedLoss

VERSION = "v2.5"

# ---- Hyperparameters -------------------------------------------------------
EPOCHS          = 400
BATCH_SIZE      = 4
LEARNING_RATE   = 1e-4
PATCH_SIZE      = 64
NUM_TAU_CH      = 8
MAX_TAU         = 64
TRAIN_RATIO     = 0.02   # 2% — 5% with 400ep caused overfitting; TV reg handles it
LAMBDA_TV       = 0.05   # Total Variation regularization weight
RANDOM_TAU      = True   # randomly sample tau delays each step
CHECKPOINT_DIR  = "./checkpoint"
RESULT_DIR      = "./result"
# ----------------------------------------------------------------------------


def train_internal_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{VERSION}] Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    print("Loading cell video...")
    import tifffile
    video_path = "./data/test_synthetic_cell.tif"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Run utils/generate_test_video.py first.")
        return

    video_matrix = tifffile.imread(video_path).astype(np.float32)
    T, H, W = video_matrix.shape
    print(f"Video shape: T={T}, H={H}, W={W}")

    train_dataset = TauSparseDataset(
        video_tensor=video_matrix,
        tau_delays=tuple(sorted(range(0, MAX_TAU + 1, MAX_TAU // (NUM_TAU_CH - 1)))[:NUM_TAU_CH]),
        patch_size=PATCH_SIZE,
        mode="train",
        train_ratio=TRAIN_RATIO,
        random_tau=RANDOM_TAU,
        num_tau_channels=NUM_TAU_CH,
        max_tau=MAX_TAU,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---- Model & Loss --------------------------------------------------------
    model     = PISSLTauEncoder(num_tau_channels=NUM_TAU_CH).to(device)
    criterion = PhysicsInformedLoss(lambda_gamma=1.0, lambda_alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- Training Loop -------------------------------------------------------
    print(f"Starting Internal Learning [{VERSION}]: {len(train_dataset)} patches/epoch, "
          f"train_ratio={TRAIN_RATIO}, random_tau={RANDOM_TAU}")

    model.train()
    history = {"loss": [], "gamma": [], "alpha": []}

    for epoch in range(EPOCHS):
        epoch_loss = epoch_gamma = epoch_alpha = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, targets, mask in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            mask    = mask.to(device)

            optimizer.zero_grad()
            preds = model(inputs)

            # Masked loss — only center 3x3 pixels are supervised
            loss, loss_gamma, loss_alpha = criterion(preds, targets, mask)

            # Total Variation regularization: penalise abrupt spatial changes.
            # Diffusion parameters are physically smooth within each cell region,
            # so TV discourages the model from overfitting individual training points.
            tv = (
                (preds[:, :, 1:, :] - preds[:, :, :-1, :]).abs().mean()
                + (preds[:, :, :, 1:] - preds[:, :, :, :-1]).abs().mean()
            )
            loss = loss + LAMBDA_TV * tv
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_gamma += loss_gamma.item()
            epoch_alpha += loss_alpha.item()
            pbar.set_postfix({
                "L":  f"{loss.item():.4f}",
                "G":  f"{loss_gamma.item():.4f}",
                "A":  f"{loss_alpha.item():.4f}",
                "TV": f"{tv.item():.4f}",
            })

        n = len(train_loader)
        scheduler.step()
        history["loss"].append(epoch_loss / n)
        history["gamma"].append(epoch_gamma / n)
        history["alpha"].append(epoch_alpha / n)
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss={epoch_loss/n:.4f}  Gamma={epoch_gamma/n:.4f}  Alpha={epoch_alpha/n:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

    # ---- Save checkpoint -----------------------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pissl_internal_{VERSION}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved → {ckpt_path}")

    # ---- Loss curve ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"],  label="Total")
    ax.plot(history["gamma"], label="Gamma")
    ax.plot(history["alpha"], label="Alpha")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Internal Learning Loss [{VERSION}]")
    ax.legend()
    loss_fig_path = os.path.join(RESULT_DIR, f"loss_curve_{VERSION}.png")
    fig.savefig(loss_fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved → {loss_fig_path}")

    # ---- Full-frame inference ------------------------------------------------
    print("\n--- Full Frame Inference ---")
    model.eval()

    inference_dataset = TauSparseDataset(
        video_tensor=video_matrix,
        tau_delays=(0, 9, 18, 27, 36, 45, 54, 64),
        mode="inference",
    )

    full_input = inference_dataset[0].unsqueeze(0).to(device)
    with torch.no_grad():
        preds_full = model(full_input)

    gamma_map = preds_full[0, 0].cpu().numpy()
    alpha_map = preds_full[0, 1].cpu().numpy()

    # Background mask via temporal CV (safety net only — model is now trained on background
    # pixels with GT=(0,0), so it should predict near-zero without this mask)
    temporal_std  = video_matrix.std(axis=0)
    temporal_mean = video_matrix.mean(axis=0)
    cv_map  = temporal_std / (temporal_mean + 1e-10)
    bg_mask = cv_map < 0.005
    gamma_map[bg_mask] = 0.0
    alpha_map[bg_mask] = 0.0
    print(f"Background pixels (CV<0.5%): {bg_mask.sum()} / {bg_mask.size}")
    print(f"Gamma map: mean={gamma_map[~bg_mask].mean():.3f}  (cell pixels only)")
    print(f"Alpha map: mean={alpha_map[~bg_mask].mean():.3f}  (cell pixels only)")

    # Load perfect GT if available for side-by-side comparison
    gt_path = "./data/test_synthetic_cell_gt.npz"
    has_gt = os.path.exists(gt_path)
    if has_gt:
        gt = np.load(gt_path)
        gt_gamma, gt_alpha = gt["gamma"], gt["alpha"]

    ncols = 4 if has_gt else 2
    fig2, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    im0 = axes[0].imshow(gamma_map, cmap="magma", vmin=0, vmax=1.0)
    axes[0].set_title(f"Pred Gamma [{VERSION}]")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(alpha_map, cmap="viridis", vmin=0, vmax=2.0)
    axes[1].set_title(f"Pred Alpha [{VERSION}]")
    plt.colorbar(im1, ax=axes[1])

    if has_gt:
        im2 = axes[2].imshow(gt_gamma, cmap="magma", vmin=0, vmax=1.0)
        axes[2].set_title("GT Gamma")
        plt.colorbar(im2, ax=axes[2])
        im3 = axes[3].imshow(gt_alpha, cmap="viridis", vmin=0, vmax=2.0)
        axes[3].set_title("GT Alpha")
        plt.colorbar(im3, ax=axes[3])

    fig2.tight_layout()
    inf_path = os.path.join(RESULT_DIR, f"inference_maps_{VERSION}.png")
    fig2.savefig(inf_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Inference maps saved → {inf_path}")


if __name__ == "__main__":
    train_internal_learning()
