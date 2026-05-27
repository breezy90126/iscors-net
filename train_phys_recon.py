"""
v3.7 — separate TV strengths for gamma and alpha.

v3.6 post-mortem:
  - tau shuffle test showed |Delta alpha|=0.130 with clearly separated spatial
    structure in the diff map — model HAS learned alpha spatial structure internally.
  - But alpha output was nearly uniform: TV (lambda=0.5 on both channels) was
    compressing the alpha dynamic range, suppressing boundary expression.
  - gamma and alpha have different physical smoothness priors:
      gamma (diffusion coefficient): physically smooth — strong TV appropriate.
      alpha (anomalous exponent): sharp region boundaries expected (e.g. heterochromatin
        vs euchromatin vs active transport zones) — weak TV needed to let boundaries form.

v3.7 fix:
  - LAMBDA_TV_GAMMA = 0.5  (unchanged — gamma is physically smooth)
  - LAMBDA_TV_ALPHA = 0.05 (10x weaker — releases alpha dynamic range)
  - Huber-TV applied separately to each output channel.

Inherited from v3.6 (unchanged):
  - tau-weighted MSE with log(tau_k) weights
  - ELU+1 for alpha output (Direction D)
  - K=10 G_norm channels input (Direction A slope channels NOT used)
  - Per-pixel normalised G_empirical input; 20% spatial blind-spot
  - (gamma, alpha) 2-channel output; amplitude removed entirely
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

VERSION = "v3.7"

# ---- Hyperparameters -------------------------------------------------------
EPOCHS          = 200
BATCH_SIZE      = 4
LEARNING_RATE   = 1e-4
PATCH_SIZE      = 64
TRAIN_FRACTION  = 0.80
RECON_TAUS      = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)   # K=10
NUM_TAU_CH      = len(RECON_TAUS)   # 10 G_norm channels
LAMBDA_TV_GAMMA = 0.5               # strong: gamma is physically smooth
LAMBDA_TV_ALPHA = 0.05              # weak: alpha has genuine sharp boundaries
TAU_WEIGHTED    = True              # log(tau_k) weights; replaces log-MSE
CHECKPOINT_DIR  = "./checkpoint"
RESULT_DIR      = "./result"
# ----------------------------------------------------------------------------


def huber_tv(x, delta=0.05):
    """Huber total variation on (B, C, H, W)."""
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]

    def _h(t):
        a = t.abs()
        return torch.where(a < delta, 0.5 * t ** 2 / delta, a - 0.5 * delta)

    return _h(dx).mean() + _h(dy).mean()


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
    model     = PISSLTauEncoder(num_tau_channels=NUM_TAU_CH,
                                predict_amplitude=False).to(device)
    criterion = PhysicsReconLoss(recon_taus=RECON_TAUS,
                                 shape_only=True,
                                 tau_weighted=TAU_WEIGHTED).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                      eta_min=1e-6)

    # ---- Training Loop -------------------------------------------------------
    print(f"[{VERSION}] {len(train_dataset)} patches/epoch  "
          f"recon_tau={RECON_TAUS}  tau_weighted={TAU_WEIGHTED}  "
          f"lambda_TV_gamma={LAMBDA_TV_GAMMA}  lambda_TV_alpha={LAMBDA_TV_ALPHA}")
    history = {"loss": [], "phys_loss": [], "tv_gamma": [], "tv_alpha": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_phys = epoch_tvg = epoch_tva = 0.0
        gamma_sum = alpha_sum = pix_count = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for g_input, g_target, train_mask in pbar:
            g_input    = g_input.to(device)
            g_target   = g_target.to(device)
            train_mask = train_mask.to(device)

            optimizer.zero_grad()
            preds = model(g_input)                               # (B, 2, P, P)

            phys_loss = criterion(preds, g_target, train_mask)
            tv_g = huber_tv(preds[:, 0:1]) if LAMBDA_TV_GAMMA > 0 else torch.tensor(0.0)
            tv_a = huber_tv(preds[:, 1:2]) if LAMBDA_TV_ALPHA > 0 else torch.tensor(0.0)
            loss = phys_loss + LAMBDA_TV_GAMMA * tv_g + LAMBDA_TV_ALPHA * tv_a

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_phys += phys_loss.item()
            epoch_tvg  += tv_g.item() if LAMBDA_TV_GAMMA > 0 else 0.0
            epoch_tva  += tv_a.item() if LAMBDA_TV_ALPHA > 0 else 0.0

            with torch.no_grad():
                m = train_mask.unsqueeze(1)
                gamma_sum += (preds[:, 0:1] * m).sum().item()
                alpha_sum += (preds[:, 1:2] * m).sum().item()
                pix_count += m.sum().item()
                mean_g = gamma_sum / (pix_count + 1e-10)
                mean_a = alpha_sum / (pix_count + 1e-10)

            pbar.set_postfix({"L": f"{loss.item():.5e}",
                              "gamma": f"{mean_g:.3f}",
                              "alpha": f"{mean_a:.3f}"})

        n = len(train_loader)
        scheduler.step()
        history["loss"].append(epoch_loss / n)
        history["phys_loss"].append(epoch_phys / n)
        history["tv_gamma"].append(epoch_tvg / n)
        history["tv_alpha"].append(epoch_tva / n)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Loss={epoch_loss/n:.5e}  Phys={epoch_phys/n:.5e}  "
              f"TV_g={epoch_tvg/n:.5e}  TV_a={epoch_tva/n:.5e}  "
              f"gamma={mean_g:.3f}  alpha={mean_a:.3f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

    # ---- Save checkpoint + loss curve ----------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pissl_phys_recon_{VERSION}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved -> {ckpt_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="Total loss")
    axes[0].plot(history["phys_loss"], label="Physics (tau-weighted MSE)", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Loss [{VERSION}]")
    axes[0].legend(); axes[0].set_yscale("log")

    axes[1].plot(history["tv_gamma"], label=f"Huber-TV gamma (lambda={LAMBDA_TV_GAMMA})", color="orange")
    axes[1].plot(history["tv_alpha"], label=f"Huber-TV alpha (lambda={LAMBDA_TV_ALPHA})", color="steelblue", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("TV loss (unscaled)")
    axes[1].set_title("TV Regularisation per Channel"); axes[1].legend()
    axes[1].set_yscale("log")

    loss_fig = os.path.join(RESULT_DIR, f"loss_curve_{VERSION}.png")
    fig.savefig(loss_fig, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Loss curve -> {loss_fig}")

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

    bg_mask = ~train_dataset.cell_mask
    gamma_map[bg_mask] = 0.0
    alpha_map[bg_mask] = 0.0
    print(f"Background pixels zeroed: {bg_mask.sum()}")

    # ---- Generalisation report -----------------------------------------------
    gt_path = "./data/test_synthetic_cell_gt.npz"
    gt_gamma = gt_alpha = None
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_gamma, gt_alpha = gt["gamma"], gt["alpha"]

        seen_mask     = train_dataset.supervised_mask.astype(bool)
        held_out_mask = train_dataset.held_out_mask

        def mae(pred, gt, mask):
            return float(np.abs(pred[mask] - gt[mask]).mean()) if mask.any() else float("nan")

        print(f"\n=== Generalisation Report [{VERSION}] ===")
        print(f"  Supervised  pixels : {seen_mask.sum()}")
        print(f"  Held-out    pixels : {held_out_mask.sum()}")
        gG_seen = mae(gamma_map, gt_gamma, seen_mask)
        gG_held = mae(gamma_map, gt_gamma, held_out_mask)
        aA_seen = mae(alpha_map, gt_alpha, seen_mask)
        aA_held = mae(alpha_map, gt_alpha, held_out_mask)
        rG = gG_held / (gG_seen + 1e-10)
        rA = aA_held / (aA_seen + 1e-10)
        print(f"  Gamma MAE -- seen    : {gG_seen:.4f}")
        print(f"  Gamma MAE -- held-out: {gG_held:.4f}  ratio: {rG:.2f}")
        print(f"  Alpha MAE -- seen    : {aA_seen:.4f}")
        print(f"  Alpha MAE -- held-out: {aA_held:.4f}  ratio: {rA:.2f}")
        print(f"  (ratio ~1 -> spatial generalisation; >>1 -> memorisation)")
        print("==============================================\n")

        rpt = os.path.join(RESULT_DIR, f"generalisation_{VERSION}.txt")
        with open(rpt, "w") as f:
            f.write(f"=== Generalisation Report [{VERSION}] ===\n")
            f.write(f"Supervised pixels : {seen_mask.sum()}\n")
            f.write(f"Held-out   pixels : {held_out_mask.sum()}\n")
            f.write(f"Gamma MAE seen    : {gG_seen:.4f}\n")
            f.write(f"Gamma MAE held-out: {gG_held:.4f}  ratio: {rG:.2f}\n")
            f.write(f"Alpha MAE seen    : {aA_seen:.4f}\n")
            f.write(f"Alpha MAE held-out: {aA_held:.4f}  ratio: {rA:.2f}\n")
        print(f"Generalisation report -> {rpt}")

    # ---- Inference maps figure -----------------------------------------------
    has_gt = gt_gamma is not None
    ncols  = 4 if has_gt else 2
    fig2, axes2 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    im0 = axes2[0].imshow(gamma_map, cmap="magma",  vmin=0, vmax=1.0)
    axes2[0].set_title(f"Pred Gamma [{VERSION}]"); plt.colorbar(im0, ax=axes2[0])
    im1 = axes2[1].imshow(alpha_map, cmap="viridis", vmin=0, vmax=2.0)
    axes2[1].set_title(f"Pred Alpha [{VERSION}]"); plt.colorbar(im1, ax=axes2[1])
    if has_gt:
        im2 = axes2[2].imshow(gt_gamma, cmap="magma",  vmin=0, vmax=1.0)
        axes2[2].set_title("GT Gamma"); plt.colorbar(im2, ax=axes2[2])
        im3 = axes2[3].imshow(gt_alpha, cmap="viridis", vmin=0, vmax=2.0)
        axes2[3].set_title("GT Alpha"); plt.colorbar(im3, ax=axes2[3])
    fig2.tight_layout()
    inf_path = os.path.join(RESULT_DIR, f"inference_maps_{VERSION}.png")
    fig2.savefig(inf_path, dpi=120, bbox_inches="tight"); plt.close(fig2)
    print(f"Inference maps -> {inf_path}")

    # ---- tau Shuffle Test ------------------------------------------------
    # Diagnostic: randomly permute the K tau channels in model input.
    # Large |delta| -> model uses tau ordering (physics curve decoding).
    # Small |delta| -> model ignores tau ordering (spatial pattern matching).
    print("\n--- tau Shuffle Test ---")
    N_SHUFFLES  = 10
    dg_list, da_list = [], []
    cell_mask_t = torch.from_numpy(train_dataset.cell_mask).to(device)

    with torch.no_grad():
        for _ in range(N_SHUFFLES):
            perm        = torch.randperm(NUM_TAU_CH)
            preds_shuf  = model(full_input[:, perm, :, :])
            dg = (preds_full[0, 0] - preds_shuf[0, 0]).abs()
            da = (preds_full[0, 1] - preds_shuf[0, 1]).abs()
            dg_list.append(dg[cell_mask_t].mean().item())
            da_list.append(da[cell_mask_t].mean().item())

    mean_dg = float(np.mean(dg_list));  std_dg = float(np.std(dg_list))
    mean_da = float(np.mean(da_list));  std_da = float(np.std(da_list))
    print(f"  |Delta gamma| ({N_SHUFFLES} shuffles): {mean_dg:.4f} +/- {std_dg:.4f}")
    print(f"  |Delta alpha| ({N_SHUFFLES} shuffles): {mean_da:.4f} +/- {std_da:.4f}")

    # gamma range (0,1), alpha range (0,2) — thresholds at 5% of each range
    if mean_dg < 0.05 and mean_da < 0.10:
        verdict = "SPATIAL — model not using tau ordering (blind-spot spatial prior dominates)"
    elif mean_dg > 0.15 or mean_da > 0.30:
        verdict = "TEMPORAL — model using tau ordering (physics curve decoding active)"
    else:
        verdict = "MIXED — partial tau sensitivity"
    print(f"  Verdict: {verdict}")

    shuf_rpt = os.path.join(RESULT_DIR, f"shuffle_test_{VERSION}.txt")
    with open(shuf_rpt, "w") as f:
        f.write(f"=== tau Shuffle Test [{VERSION}] ===\n")
        f.write(f"N shuffles        : {N_SHUFFLES}\n")
        f.write(f"|Delta gamma| mean: {mean_dg:.4f}  std: {std_dg:.4f}\n")
        f.write(f"|Delta alpha| mean: {mean_da:.4f}  std: {std_da:.4f}\n")
        f.write(f"Verdict           : {verdict}\n")
    print(f"Shuffle report -> {shuf_rpt}")

    # Visualise one shuffle: original | shuffled | |diff|
    with torch.no_grad():
        perm_vis   = torch.randperm(NUM_TAU_CH)
        preds_vis  = model(full_input[:, perm_vis, :, :])
    gm_shuf = preds_vis[0, 0].cpu().numpy(); gm_shuf[bg_mask] = 0.0
    am_shuf = preds_vis[0, 1].cpu().numpy(); am_shuf[bg_mask] = 0.0
    diff_g  = np.abs(gamma_map - gm_shuf)
    diff_a  = np.abs(alpha_map - am_shuf)

    fig3, ax3 = plt.subplots(2, 3, figsize=(15, 8))
    for row, (orig, shuf, diff, cmap, vmax, lbl) in enumerate([
        (gamma_map, gm_shuf, diff_g, "magma",   1.0, "Gamma"),
        (alpha_map, am_shuf, diff_a, "viridis", 2.0, "Alpha"),
    ]):
        cell = train_dataset.cell_mask
        im = ax3[row,0].imshow(orig, cmap=cmap, vmin=0, vmax=vmax)
        ax3[row,0].set_title(f"{lbl} original"); plt.colorbar(im, ax=ax3[row,0])
        im = ax3[row,1].imshow(shuf, cmap=cmap, vmin=0, vmax=vmax)
        ax3[row,1].set_title(f"{lbl} shuffled-tau"); plt.colorbar(im, ax=ax3[row,1])
        im = ax3[row,2].imshow(diff, cmap="hot", vmin=0)
        ax3[row,2].set_title(f"|diff| {lbl}  mean={diff[cell].mean():.3f}")
        plt.colorbar(im, ax=ax3[row,2])
    fig3.suptitle(f"tau Shuffle Test [{VERSION}]  perm={perm_vis.tolist()}")
    fig3.tight_layout()
    shuf_fig = os.path.join(RESULT_DIR, f"shuffle_test_{VERSION}.png")
    fig3.savefig(shuf_fig, dpi=120, bbox_inches="tight"); plt.close(fig3)
    print(f"Shuffle test figure -> {shuf_fig}")


if __name__ == "__main__":
    train_physics_reconstruction()
