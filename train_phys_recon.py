"""
v3.3 — Spatial blind-spot Physics Reconstruction training.

Changes vs v3.2:
  • Model input is masked G_empirical(τ) map instead of raw tau-slices.
    - 80% of cell pixels: G_empirical visible in input, loss applied.
    - 20% of cell pixels: G_empirical zeroed in input, excluded from loss.
    Model must infer (γ,α,A) at held-out pixels from spatial neighbours
    (spatial redundancy / FAST-style blind-spot on G maps).
  • This restores a meaningful 80/20 generalization test: held-out pixels
    are truly hidden from the model, unlike v3.0–v3.2 where raw-frame input
    still contained temporal info from all pixels.
  • Input channels = K = len(RECON_TAUS) (unchanged at 8).
  • Loss, model architecture, and amplitude output unchanged from v3.2.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from datasets.phys_recon_dataset import PhysReconDataset
from models.pissl_tau_encoder import PISSLTauEncoder
from loss.phys_recon_loss import PhysicsReconLoss

VERSION = "v3.3"

# ---- Hyperparameters -------------------------------------------------------
EPOCHS          = 200
BATCH_SIZE      = 4
LEARNING_RATE   = 1e-4
PATCH_SIZE      = 64
TRAIN_FRACTION  = 0.80           # 80/20 hold-out split on cell pixels
LAMBDA_TV       = 0.00           # Huber-TV off
HUBER_DELTA     = 0.05
LOG_SPACE       = False          # flip to True for log-MSE
RECON_TAUS      = (1, 2, 4, 8, 16, 32, 48, 64)
NUM_TAU_CH      = len(RECON_TAUS)   # input channels = K (masked G channels)
CHECKPOINT_DIR  = "./checkpoint"
RESULT_DIR      = "./result"
# ----------------------------------------------------------------------------


def huber_tv(preds, delta=HUBER_DELTA):
    gy = preds[:, :, 1:, :] - preds[:, :, :-1, :]
    gx = preds[:, :, :, 1:] - preds[:, :, :, :-1]

    def h(g):
        absg = g.abs()
        return torch.where(absg < delta, 0.5 * g.pow(2) / delta, absg - 0.5 * delta)
    return h(gy).mean() + h(gx).mean()


def train_physics_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{VERSION}] Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    import tifffile
    video_path = "./data/test_synthetic_cell.tif"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return
    video_matrix = tifffile.imread(video_path).astype(np.float32)
    T, H, W = video_matrix.shape
    print(f"Video shape: T={T}, H={H}, W={W}")

    train_dataset = PhysReconDataset(
        video_tensor=video_matrix,
        recon_taus=RECON_TAUS,
        patch_size=PATCH_SIZE,
        mode="train",
        train_fraction=TRAIN_FRACTION,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---- Model & Loss --------------------------------------------------------
    model     = PISSLTauEncoder(num_tau_channels=NUM_TAU_CH, predict_amplitude=True).to(device)
    criterion = PhysicsReconLoss(recon_taus=RECON_TAUS, log_space=LOG_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- Training Loop -------------------------------------------------------
    print(f"Starting Physics Reconstruction [{VERSION}]: "
          f"{len(train_dataset)} supervised patches/epoch, "
          f"recon_τ={RECON_TAUS}, train_fraction={TRAIN_FRACTION}, "
          f"log_space={LOG_SPACE}, λ_TV={LAMBDA_TV}")
    history = {"loss": [], "tv": [], "phys": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_phys = epoch_tv = 0.0
        gamma_sum = alpha_sum = amp_sum = pix_count = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, g_empirical, train_mask in pbar:
            inputs       = inputs.to(device)
            g_empirical  = g_empirical.to(device)
            train_mask   = train_mask.to(device)

            optimizer.zero_grad()
            preds = model(inputs)                        # (B, 3, P, P)

            loss_phys = criterion(preds, g_empirical, train_mask)
            if LAMBDA_TV > 0:
                loss_tv = huber_tv(preds)
                loss = loss_phys + LAMBDA_TV * loss_tv
                epoch_tv += loss_tv.item()
            else:
                loss = loss_phys
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_phys += loss_phys.item()

            # Mean predicted (γ, α, A) on supervised pixels — detects collapse
            with torch.no_grad():
                m = train_mask.unsqueeze(1)  # (B,1,P,P)
                gamma_sum += (preds[:, 0:1] * m).sum().item()
                alpha_sum += (preds[:, 1:2] * m).sum().item()
                amp_sum   += (preds[:, 2:3] * m).sum().item()
                pix_count += m.sum().item()
                mean_g = gamma_sum / (pix_count + 1e-10)
                mean_a = alpha_sum / (pix_count + 1e-10)
                mean_A = amp_sum   / (pix_count + 1e-10)

            pbar.set_postfix({
                "L":   f"{loss.item():.5f}",
                "γ̄": f"{mean_g:.3f}",
                "ᾱ": f"{mean_a:.3f}",
                "Ā":  f"{mean_A:.2e}",
            })

        n = len(train_loader)
        scheduler.step()
        history["loss"].append(epoch_loss / n)
        history["phys"].append(epoch_phys / n)
        history["tv"].append(epoch_tv / n)
        mean_g = gamma_sum / (pix_count + 1e-10)
        mean_a = alpha_sum / (pix_count + 1e-10)
        mean_A = amp_sum   / (pix_count + 1e-10)
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss={epoch_loss/n:.5e}  Phys={epoch_phys/n:.5e}  "
              f"γ̄={mean_g:.3f}  ᾱ={mean_a:.3f}  Ā={mean_A:.2e}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

    # ---- Save checkpoint + loss curve ----------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pissl_phys_recon_{VERSION}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved → {ckpt_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Total")
    ax.plot(history["phys"], label="Phys recon")
    if LAMBDA_TV > 0:
        ax.plot(history["tv"], label="Huber-TV")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Physics Reconstruction Loss [{VERSION}]")
    ax.legend()
    ax.set_yscale("log")
    loss_fig_path = os.path.join(RESULT_DIR, f"loss_curve_{VERSION}.png")
    fig.savefig(loss_fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved → {loss_fig_path}")

    # ---- Full-frame inference ------------------------------------------------
    print("\n--- Full Frame Inference ---")
    model.eval()
    infer_dataset = PhysReconDataset(
        video_tensor=video_matrix,
        recon_taus=RECON_TAUS,
        patch_size=PATCH_SIZE,
        mode="inference",
    )
    full_input = infer_dataset[0].unsqueeze(0).to(device)
    with torch.no_grad():
        preds_full = model(full_input)

    gamma_map = preds_full[0, 0].cpu().numpy()
    alpha_map = preds_full[0, 1].cpu().numpy()
    amp_map   = preds_full[0, 2].cpu().numpy()

    # Background safety net: zero out near-static pixels
    bg_mask = ~train_dataset.cell_mask
    gamma_map[bg_mask] = 0.0
    alpha_map[bg_mask] = 0.0
    amp_map[bg_mask]   = 0.0
    print(f"Background pixels: {bg_mask.sum()} / {bg_mask.size}")
    print(f"Predicted A — cell  : mean={amp_map[~bg_mask].mean():.3e}  "
          f"min={amp_map[~bg_mask].min():.3e}  max={amp_map[~bg_mask].max():.3e}")

    # ---- Generalisation evaluation (held-out 20%) ----------------------------
    gt_path = "./data/test_synthetic_cell_gt.npz"
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_gamma, gt_alpha = gt["gamma"], gt["alpha"]

        seen_mask     = train_dataset.supervised_mask.astype(bool)
        held_out_mask = train_dataset.held_out_mask

        def mae(pred, gt, mask):
            return float(np.abs(pred[mask] - gt[mask]).mean()) if mask.any() else float("nan")

        print(f"\n=== Generalisation Report [{VERSION}] ===")
        print(f"  Supervised (80%) pixels : {seen_mask.sum()}")
        print(f"  Held-out  (20%) pixels  : {held_out_mask.sum()}")
        gG_seen = mae(gamma_map, gt_gamma, seen_mask)
        gG_held = mae(gamma_map, gt_gamma, held_out_mask)
        aA_seen = mae(alpha_map, gt_alpha, seen_mask)
        aA_held = mae(alpha_map, gt_alpha, held_out_mask)
        print(f"  Gamma MAE — seen    : {gG_seen:.4f}")
        print(f"  Gamma MAE — held-out: {gG_held:.4f}")
        print(f"  Alpha MAE — seen    : {aA_seen:.4f}")
        print(f"  Alpha MAE — held-out: {aA_held:.4f}")
        rG = gG_held / (gG_seen + 1e-10)
        rA = aA_held / (aA_seen + 1e-10)
        print(f"  Held-out/Seen ratio — Gamma: {rG:.2f}   Alpha: {rA:.2f}")
        print(f"  (ratio ≈ 1 → generalises; >> 1 → memorises supervised set)")
        print("====================================================\n")

        report_path = os.path.join(RESULT_DIR, f"generalisation_{VERSION}.txt")
        with open(report_path, "w") as f:
            f.write(f"=== Generalisation Report [{VERSION}] (amplitude-aware) ===\n")
            f.write(f"Supervised (80%) pixels: {seen_mask.sum()}\n")
            f.write(f"Held-out  (20%) pixels : {held_out_mask.sum()}\n")
            f.write(f"Gamma MAE — seen      : {gG_seen:.4f}\n")
            f.write(f"Gamma MAE — held-out  : {gG_held:.4f}\n")
            f.write(f"Alpha MAE — seen      : {aA_seen:.4f}\n")
            f.write(f"Alpha MAE — held-out  : {aA_held:.4f}\n")
            f.write(f"Held-out/Seen ratio — Gamma: {rG:.2f}  Alpha: {rA:.2f}\n")
            f.write(f"Pred A (cell): mean={amp_map[~bg_mask].mean():.3e}  "
                    f"min={amp_map[~bg_mask].min():.3e}  max={amp_map[~bg_mask].max():.3e}\n")
        print(f"Generalisation report saved → {report_path}")
    else:
        gt_gamma = gt_alpha = None

    # ---- Inference maps figure -----------------------------------------------
    has_gt = gt_gamma is not None
    # Layout: row 1 = [γ_pred, α_pred, A_pred]; row 2 = [γ_gt, α_gt, blank] if GT present
    if has_gt:
        fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
        im0 = axes[0, 0].imshow(gamma_map, cmap="magma", vmin=0, vmax=1.0)
        axes[0, 0].set_title(f"Pred Gamma [{VERSION}]")
        plt.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(alpha_map, cmap="viridis", vmin=0, vmax=2.0)
        axes[0, 1].set_title(f"Pred Alpha [{VERSION}]")
        plt.colorbar(im1, ax=axes[0, 1])
        im2 = axes[0, 2].imshow(amp_map, cmap="cividis")
        axes[0, 2].set_title(f"Pred Amplitude [{VERSION}]")
        plt.colorbar(im2, ax=axes[0, 2])
        im3 = axes[1, 0].imshow(gt_gamma, cmap="magma", vmin=0, vmax=1.0)
        axes[1, 0].set_title("GT Gamma")
        plt.colorbar(im3, ax=axes[1, 0])
        im4 = axes[1, 1].imshow(gt_alpha, cmap="viridis", vmin=0, vmax=2.0)
        axes[1, 1].set_title("GT Alpha")
        plt.colorbar(im4, ax=axes[1, 1])
        axes[1, 2].axis("off")
    else:
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(gamma_map, cmap="magma", vmin=0, vmax=1.0)
        axes[0].set_title(f"Pred Gamma [{VERSION}]")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(alpha_map, cmap="viridis", vmin=0, vmax=2.0)
        axes[1].set_title(f"Pred Alpha [{VERSION}]")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(amp_map, cmap="cividis")
        axes[2].set_title(f"Pred Amplitude [{VERSION}]")
        plt.colorbar(im2, ax=axes[2])
    fig2.tight_layout()
    inf_path = os.path.join(RESULT_DIR, f"inference_maps_{VERSION}.png")
    fig2.savefig(inf_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Inference maps saved → {inf_path}")


if __name__ == "__main__":
    train_physics_reconstruction()
