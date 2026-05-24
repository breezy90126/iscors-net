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

VERSION = "v2.7"

# ---- Hyperparameters -------------------------------------------------------
EPOCHS          = 400
BATCH_SIZE      = 4
LEARNING_RATE   = 1e-4
PATCH_SIZE      = 64
NUM_TAU_CH      = 8
MAX_TAU         = 64
TRAIN_RATIO     = 0.10   # 10% sparse pixels (TV + weight_decay handle overfitting)
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

    # Use perfect GT from simulation when available (synthetic validation).
    # For real data (no _gt.npz), set gt_npz_path=None to use curve fitting.
    gt_npz_path = video_path.replace(".tif", "_gt.npz")
    if os.path.exists(gt_npz_path):
        print(f"Perfect GT found: {gt_npz_path} — skipping curve fitting.")
    else:
        gt_npz_path = None
        print("No perfect GT found — will estimate GT via curve fitting.")

    train_dataset = TauSparseDataset(
        video_tensor=video_matrix,
        tau_delays=tuple(sorted(range(0, MAX_TAU + 1, MAX_TAU // (NUM_TAU_CH - 1)))[:NUM_TAU_CH]),
        patch_size=PATCH_SIZE,
        mode="train",
        train_ratio=TRAIN_RATIO,
        random_tau=RANDOM_TAU,
        num_tau_channels=NUM_TAU_CH,
        max_tau=MAX_TAU,
        gt_npz_path=gt_npz_path,
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

    # Load perfect GT if available for quantitative evaluation + side-by-side comparison
    gt_path = "./data/test_synthetic_cell_gt.npz"
    has_gt = os.path.exists(gt_path)
    if has_gt:
        gt = np.load(gt_path)
        gt_gamma, gt_alpha = gt["gamma"], gt["alpha"]

        # ---- Seen vs Unseen generalisation check --------------------------------
        # Build a boolean mask of which pixels were used as training supervision.
        # If the model only memorised training points, seen_mae << unseen_mae.
        # If it learned the autocorrelation physics, both should be comparable.
        seen_mask = np.zeros((H, W), dtype=bool)
        for (y, x) in train_dataset.train_coords:
            seen_mask[y, x] = True
        cell_mask = ~bg_mask  # evaluate on cell pixels only (BG is trivially zero)

        seen_cell   = seen_mask & cell_mask
        unseen_cell = (~seen_mask) & cell_mask

        def mae(pred, gt, mask):
            return np.abs(pred[mask] - gt[mask]).mean() if mask.any() else float("nan")

        print("\n=== Generalisation Report ===")
        print(f"  Training pixels (seen)  : {seen_cell.sum()}")
        print(f"  Unseen cell pixels      : {unseen_cell.sum()}")
        print(f"  Gamma MAE — seen   : {mae(gamma_map, gt_gamma, seen_cell):.4f}")
        print(f"  Gamma MAE — unseen : {mae(gamma_map, gt_gamma, unseen_cell):.4f}")
        print(f"  Alpha MAE — seen   : {mae(alpha_map, gt_alpha, seen_cell):.4f}")
        print(f"  Alpha MAE — unseen : {mae(alpha_map, gt_alpha, unseen_cell):.4f}")
        ratio_g = mae(gamma_map, gt_gamma, unseen_cell) / (mae(gamma_map, gt_gamma, seen_cell) + 1e-10)
        ratio_a = mae(alpha_map, gt_alpha, unseen_cell) / (mae(alpha_map, gt_alpha, seen_cell) + 1e-10)
        print(f"  Unseen/Seen ratio — Gamma: {ratio_g:.2f}  Alpha: {ratio_a:.2f}")
        print(f"  (ratio ≈ 1 → model generalises; ratio >> 1 → memorisation)")
        print("=============================\n")

        # Save generalisation report as text
        report_path = os.path.join(RESULT_DIR, f"generalisation_{VERSION}.txt")
        with open(report_path, "w") as f:
            f.write(f"=== Generalisation Report [{VERSION}] ===\n")
            f.write(f"Training pixels (seen)  : {seen_cell.sum()}\n")
            f.write(f"Unseen cell pixels      : {unseen_cell.sum()}\n")
            f.write(f"Gamma MAE — seen   : {mae(gamma_map, gt_gamma, seen_cell):.4f}\n")
            f.write(f"Gamma MAE — unseen : {mae(gamma_map, gt_gamma, unseen_cell):.4f}\n")
            f.write(f"Alpha MAE — seen   : {mae(alpha_map, gt_alpha, seen_cell):.4f}\n")
            f.write(f"Alpha MAE — unseen : {mae(alpha_map, gt_alpha, unseen_cell):.4f}\n")
            f.write(f"Unseen/Seen ratio — Gamma: {ratio_g:.2f}  Alpha: {ratio_a:.2f}\n")
            f.write("ratio ≈ 1 → generalises to unseen pixels\n")
            f.write("ratio >> 1 → memorisation, not learning\n")
        print(f"Generalisation report saved → {report_path}")

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
