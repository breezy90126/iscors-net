"""
v4.0 — Masked Huber-TV (cell-only pairs) + α TV ×3 boost.

v3.9 post-mortem:
  - τ-PE (v3.9 Dir-1): SUCCESS. Shuffle diff pattern became spatially uniform
    (not ring). |Δα| = 0.393 (highest so far). τ-PE broke bag-of-values shortcut.
  - 35% blind-spot (v3.9 Dir-2): SUCCESS. held-out MAE < seen MAE for both
    parameters (held-out benefits from smooth spatial propagation from trained
    neighbours). Spatial physics decoding strengthened.
  - γ collapse diagnosed: cell body (γ=0.1) and slow spot (γ=0.05) both
    predicted near-zero. Fast spot (γ=0.5) correctly identified.
    Root cause: huber_tv was applied to the FULL patch (B,1,P,P), including
    background pixels. Background pixels receive G_norm=0 input → model learns
    γ≈0 for them (no physics loss). TV across background-cell boundaries then
    pulls cell-edge γ toward 0, cascading inward → γ collapse for small-γ regions.
  - TV_alpha plateau at 0.1 throughout 200 epochs: same cause. The large
    background-cell boundary differences dominated the mean TV value, making
    within-region α smoothing ineffective relative to the floor.

v4.0 fixes:

  Fix 1 — Masked Huber-TV (cell-cell pairs only):
    Replace huber_tv(preds) with masked_huber_tv(preds, cell_mask_patch).
    Only adjacent pixel pairs where BOTH pixels are cell (CV ≥ 0.005) contribute.
    This eliminates the background→cell cascade that caused γ collapse.

    Mechanism:
      Before: TV penalises |γ_cell_edge − γ_background| ≈ |0.1 − 0|. TV wins
              over physics loss for small γ → pulls edge toward 0 → propagates
              inward via TV chain across cell interior.
      After:  Background-cell pairs excluded from TV. Physics loss drives γ toward
              true value at cell pixels. TV smooths only within-cell spatial noise.

  Fix 2 — α TV ×3 boost (TV_ALPHA_EXTRA_SCALE):
    With masking, the background-cell boundary floor that inflated TV_alpha is
    removed. The remaining within-cell TV_alpha is smaller, so λ_TV_alpha needs
    upward adjustment to maintain similar regularisation strength.
    TV_ALPHA_EXTRA_SCALE = 3.0 → effective λ_TV_alpha = 3.54e-3 × 3 = 1.06e-2.
    This makes TV contribution ~10% of physics loss, which is the target range.

Inherited unchanged from v3.9:
  - τ positional encoding (K extra channels, log(τ)/log(τ_max))
  - 35% blind-spot (TRAIN_FRACTION = 0.65)
  - ELU+1 for alpha output (Direction D)
  - Fisher information τ-weighting (peaked at τ=8–16)
  - PHYSICS_TV_CORRECTION = 10.0 (kept; masked TV slightly lowers effective
    magnitude so correction is still appropriate)
  - Per-pixel normalised G_empirical input; (gamma, alpha) 2-channel output
  - τ shuffle test diagnostic
  - Cross-video overfitting test (v1 train → v2 inference)
"""

import math
import os

import numpy as np
import torch
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.phys_recon_dataset import PhysReconDataset
from models.pissl_tau_encoder import PISSLTauEncoder
from loss.phys_recon_loss import PhysicsReconLoss

VERSION = "v4.0"

# ---- Training hyperparameters -----------------------------------------------
EPOCHS         = 200
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-4
PATCH_SIZE     = 64
TRAIN_FRACTION = 0.65                                       # v3.9: 35% blind-spot
RECON_TAUS     = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)   # K=10
NUM_TAU_CH     = len(RECON_TAUS)
CHECKPOINT_DIR = "./checkpoint"
RESULT_DIR     = "./result"

# ---- Optical parameters for physics-derived TV ------------------------------
# Adjust these to match the actual microscope setup.
WAVELENGTH_NM       = 532.0   # iSCAT illumination wavelength (nm)
NA                  = 1.4     # objective numerical aperture
PIXEL_SIZE_NM       = 65.0    # pixel size at sample plane (nm)
#   PSF Rayleigh limit = 0.61 * λ / NA / pixel_size  (in pixels)
#   For 532nm, NA=1.4, 65nm/px: L_PSF ≈ 3.57 px
# v3.9 correction: physics floor underestimates needed strength by ~10×
# (initialisation noise not accounted for in pure noise-floor derivation)
PHYSICS_TV_CORRECTION = 10.0

# v4.0: boost α TV by extra ×3 to compensate for smaller effective TV after
# background-cell boundary pairs are excluded by masked_huber_tv.
# Target: λ_TV_alpha × TV_alpha ≈ 10% of physics_loss magnitude.
TV_ALPHA_EXTRA_SCALE  = 3.0

# ---- Fisher information τ-weighting prior -----------------------------------
# Evaluated at (γ₀, α₀) representing a "typical cell pixel" (not from GT).
# Stable training: use fixed prior, not per-pixel dynamic weights.
FISHER_GAMMA_PRIOR = 0.1   # typical cell body γ (iSCAT high-speed)
FISHER_ALPHA_PRIOR = 1.0   # baseline normal diffusion α
# -----------------------------------------------------------------------------


def compute_physics_tv_lambdas(T, wavelength_nm, na, pixel_nm):
    """
    Derive λ_TV for γ and α from physical quantities only.

    Formula: λ_TV = σ²_G / (σ²_prior × L²_PSF)

      σ²_G    = 1/T                — G_norm estimation noise floor
      L_PSF   = 0.61·λ/NA/pixel   — Rayleigh limit (pixels)
      σ²_prior_γ = (γ_max/6)²     — 6-sigma prior over γ ∈ [0, 1]  (Sigmoid bound)
      σ²_prior_α = (α_max/6)²     — 6-sigma prior over α ∈ [0, 2]  (ELU+1 bound)

    The parameter ranges come from model activation bounds, not GT knowledge.
    Ratio λ_γ/λ_α = σ²_prior_α/σ²_prior_γ = (α_max/γ_max)² = 4 — physics says
    α has larger dynamic range, so its TV should be proportionally weaker.
    """
    L_psf   = 0.61 * wavelength_nm / na / pixel_nm          # Rayleigh (pixels)
    sigma2G = 1.0 / T                                        # noise floor
    lam_g   = sigma2G / ((1.0 / 6) ** 2 * L_psf ** 2)       # γ_max=1
    lam_a   = sigma2G / ((2.0 / 6) ** 2 * L_psf ** 2)       # α_max=2
    return lam_g * PHYSICS_TV_CORRECTION, lam_a * PHYSICS_TV_CORRECTION, L_psf


def masked_huber_tv(x, cell_mask, delta=0.05):
    """Huber TV restricted to adjacent pairs where BOTH pixels are cell pixels.

    Excluding background-cell boundary pairs prevents TV from pulling cell-edge
    values toward the background value (typically γ≈0), which caused γ collapse
    in v3.9 via a cascade from boundary into the cell interior.

    Args:
        x         : (B, 1, H, W) — γ or α prediction map.
        cell_mask : (B, H, W) float — 1 at cell pixels (CV ≥ min_cv), 0 elsewhere.
    """
    m  = cell_mask.unsqueeze(1).float()          # (B, 1, H, W)
    dx = x[..., 1:] - x[..., :-1]               # (B, 1, H, W-1)
    dy = x[..., 1:, :] - x[..., :-1, :]         # (B, 1, H-1, W)
    mx = m[..., 1:] * m[..., :-1]               # 1 where both neighbours are cell
    my = m[..., 1:, :] * m[..., :-1, :]

    def _h(t):
        a = t.abs()
        return torch.where(a < delta, 0.5 * t ** 2 / delta, a - 0.5 * delta)

    n = mx.sum() + my.sum() + 1e-10
    return (_h(dx) * mx).sum() / n + (_h(dy) * my).sum() / n


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

    # ---- Physics-derived TV lambdas (computed from T and optics) -------------
    LAMBDA_TV_GAMMA, LAMBDA_TV_ALPHA, L_psf = compute_physics_tv_lambdas(
        T, WAVELENGTH_NM, NA, PIXEL_SIZE_NM
    )
    print(f"[{VERSION}] PSF = {L_psf:.3f} px  "
          f"(λ={WAVELENGTH_NM}nm, NA={NA}, pixel={PIXEL_SIZE_NM}nm)")
    print(f"[{VERSION}] Physics-derived TV:  "
          f"λ_gamma={LAMBDA_TV_GAMMA:.3e}  λ_alpha={LAMBDA_TV_ALPHA:.3e}  "
          f"ratio={LAMBDA_TV_GAMMA/LAMBDA_TV_ALPHA:.1f}×")

    train_dataset = PhysReconDataset(
        video_tensor=video_matrix,
        recon_taus=RECON_TAUS,
        patch_size=PATCH_SIZE,
        mode="train",
        train_fraction=TRAIN_FRACTION,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---- Model & Loss --------------------------------------------------------
    model     = PISSLTauEncoder(recon_taus=RECON_TAUS,
                                predict_amplitude=False).to(device)
    criterion = PhysicsReconLoss(
        recon_taus=RECON_TAUS,
        shape_only=True,
        fisher_weighted=True,
        fisher_gamma_prior=FISHER_GAMMA_PRIOR,
        fisher_alpha_prior=FISHER_ALPHA_PRIOR,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                      eta_min=1e-6)

    # Print Fisher weight profile for transparency
    fw = criterion.fisher_weights.squeeze().cpu().tolist()
    print(f"[{VERSION}] Fisher τ-weights at (γ₀={FISHER_GAMMA_PRIOR}, α₀={FISHER_ALPHA_PRIOR}):")
    print("  " + "  ".join(f"τ={t}:{w:.3f}" for t, w in zip(RECON_TAUS, fw)))

    # ---- Training Loop -------------------------------------------------------
    print(f"[{VERSION}] {len(train_dataset)} patches/epoch  "
          f"fisher_weighted=True  masked_TV=True  "
          f"λ_TV_gamma={LAMBDA_TV_GAMMA:.3e}  "
          f"λ_TV_alpha={LAMBDA_TV_ALPHA * TV_ALPHA_EXTRA_SCALE:.3e} "
          f"(={LAMBDA_TV_ALPHA:.3e} × {TV_ALPHA_EXTRA_SCALE})")
    history = {"loss": [], "phys_loss": [], "tv_gamma": [], "tv_alpha": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_phys = epoch_tvg = epoch_tva = 0.0
        gamma_sum = alpha_sum = pix_count = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for g_input, g_target, train_mask, cell_mask_patch in pbar:
            g_input          = g_input.to(device)
            g_target         = g_target.to(device)
            train_mask       = train_mask.to(device)
            cell_mask_patch  = cell_mask_patch.to(device)

            optimizer.zero_grad()
            preds = model(g_input)                               # (B, 2, P, P)

            phys_loss = criterion(preds, g_target, train_mask)
            # Masked TV: only cell-cell adjacent pairs (excludes background-cell
            # boundaries that previously caused γ collapse via pulling cascade).
            tv_g = masked_huber_tv(preds[:, 0:1], cell_mask_patch) \
                   if LAMBDA_TV_GAMMA > 0 else torch.tensor(0.0)
            tv_a = masked_huber_tv(preds[:, 1:2], cell_mask_patch) \
                   if LAMBDA_TV_ALPHA > 0 else torch.tensor(0.0)
            loss = (phys_loss
                    + LAMBDA_TV_GAMMA * tv_g
                    + LAMBDA_TV_ALPHA * TV_ALPHA_EXTRA_SCALE * tv_a)

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
                              "γ": f"{mean_g:.3f}", "α": f"{mean_a:.3f}"})

        n = len(train_loader)
        scheduler.step()
        history["loss"].append(epoch_loss / n)
        history["phys_loss"].append(epoch_phys / n)
        history["tv_gamma"].append(epoch_tvg / n)
        history["tv_alpha"].append(epoch_tva / n)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Loss={epoch_loss/n:.5e}  Phys={epoch_phys/n:.5e}  "
              f"TV_g={epoch_tvg/n:.5e}  TV_a={epoch_tva/n:.5e}  "
              f"γ={mean_g:.3f}  α={mean_a:.3f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

    # ---- Save checkpoint + loss curve ----------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pissl_phys_recon_{VERSION}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved -> {ckpt_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="Total loss")
    axes[0].plot(history["phys_loss"], label="Physics (Fisher-weighted MSE)", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Loss [{VERSION}]")
    axes[0].legend(); axes[0].set_yscale("log")

    axes[1].plot(history["tv_gamma"],
                 label=f"Masked-TV γ  λ={LAMBDA_TV_GAMMA:.2e}", color="orange")
    axes[1].plot(history["tv_alpha"],
                 label=f"Masked-TV α  λ_eff={LAMBDA_TV_ALPHA*TV_ALPHA_EXTRA_SCALE:.2e}",
                 color="steelblue", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("TV loss (unscaled)")
    axes[1].set_title(f"TV Regularisation (physics-derived λ)"); axes[1].legend()
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
    gt_path  = "./data/test_synthetic_cell_gt.npz"
    gt_gamma = gt_alpha = None
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_gamma, gt_alpha = gt["gamma"], gt["alpha"]

        seen_mask     = train_dataset.supervised_mask.astype(bool)
        held_out_mask = train_dataset.held_out_mask

        def mae(pred, gt_map, mask):
            return float(np.abs(pred[mask] - gt_map[mask]).mean()) if mask.any() else float("nan")

        gG_seen = mae(gamma_map, gt_gamma, seen_mask)
        gG_held = mae(gamma_map, gt_gamma, held_out_mask)
        aA_seen = mae(alpha_map, gt_alpha, seen_mask)
        aA_held = mae(alpha_map, gt_alpha, held_out_mask)
        rG = gG_held / (gG_seen + 1e-10)
        rA = aA_held / (aA_seen + 1e-10)

        print(f"\n=== Generalisation Report [{VERSION}] ===")
        print(f"  Supervised  pixels : {seen_mask.sum()}")
        print(f"  Held-out    pixels : {held_out_mask.sum()}")
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
            f.write(f"PSF_px            : {L_psf:.3f}\n")
            f.write(f"lambda_TV_gamma   : {LAMBDA_TV_GAMMA:.3e}\n")
            f.write(f"lambda_TV_alpha   : {LAMBDA_TV_ALPHA:.3e}\n")
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
    print("\n--- tau Shuffle Test ---")
    N_SHUFFLES  = 10
    dg_list, da_list = [], []
    cell_mask_t = torch.from_numpy(train_dataset.cell_mask).to(device)

    with torch.no_grad():
        for _ in range(N_SHUFFLES):
            perm       = torch.randperm(NUM_TAU_CH)
            preds_shuf = model(full_input[:, perm, :, :])
            dg = (preds_full[0, 0] - preds_shuf[0, 0]).abs()
            da = (preds_full[0, 1] - preds_shuf[0, 1]).abs()
            dg_list.append(dg[cell_mask_t].mean().item())
            da_list.append(da[cell_mask_t].mean().item())

    mean_dg = float(np.mean(dg_list)); std_dg = float(np.std(dg_list))
    mean_da = float(np.mean(da_list)); std_da = float(np.std(da_list))
    print(f"  |Δgamma| ({N_SHUFFLES} shuffles): {mean_dg:.4f} ± {std_dg:.4f}")
    print(f"  |Δalpha| ({N_SHUFFLES} shuffles): {mean_da:.4f} ± {std_da:.4f}")

    if mean_dg < 0.05 and mean_da < 0.10:
        verdict = "SPATIAL — model not using tau ordering"
    elif mean_dg > 0.15 or mean_da > 0.30:
        verdict = "TEMPORAL — physics curve decoding active"
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

    with torch.no_grad():
        perm_vis  = torch.randperm(NUM_TAU_CH)
        preds_vis = model(full_input[:, perm_vis, :, :])
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
        im = ax3[row, 0].imshow(orig, cmap=cmap, vmin=0, vmax=vmax)
        ax3[row, 0].set_title(f"{lbl} original"); plt.colorbar(im, ax=ax3[row, 0])
        im = ax3[row, 1].imshow(shuf, cmap=cmap, vmin=0, vmax=vmax)
        ax3[row, 1].set_title(f"{lbl} shuffled-τ"); plt.colorbar(im, ax=ax3[row, 1])
        im = ax3[row, 2].imshow(diff, cmap="hot", vmin=0)
        ax3[row, 2].set_title(f"|diff| {lbl}  mean={diff[cell].mean():.3f}")
        plt.colorbar(im, ax=ax3[row, 2])
    fig3.suptitle(f"τ Shuffle Test [{VERSION}]  perm={perm_vis.tolist()}")
    fig3.tight_layout()
    shuf_fig = os.path.join(RESULT_DIR, f"shuffle_test_{VERSION}.png")
    fig3.savefig(shuf_fig, dpi=120, bbox_inches="tight"); plt.close(fig3)
    print(f"Shuffle test figure -> {shuf_fig}")

    # ---- Cross-video overfitting test (v2 geometry) --------------------------
    # Model trained on v1 (nested circles, hard edges).
    # Run inference on v2 (concentric rings, soft edges, different parameters).
    # Consistent MAE → method generalises. Poor MAE → spatial memorisation.
    v2_path = "./data/test_synthetic_cell_v2.tif"
    if not os.path.exists(v2_path):
        print(f"\n[v2 overfitting test] {v2_path} not found — skipping.")
        print("  Generate it with: python utils/generate_test_video.py")
    else:
        print("\n--- Cross-video Overfitting Test (v2 geometry) ---")
        import tifffile as _tff
        v2_matrix = _tff.imread(v2_path).astype(np.float32)
        print(f"  v2 video shape: {v2_matrix.shape}")

        v2_infer = PhysReconDataset(
            video_tensor=v2_matrix,
            recon_taus=RECON_TAUS,
            patch_size=PATCH_SIZE,
            mode="inference",
        )
        v2_input = v2_infer[0].unsqueeze(0).to(device)
        with torch.no_grad():
            v2_preds = model(v2_input)

        v2_gamma = v2_preds[0, 0].cpu().numpy()
        v2_alpha = v2_preds[0, 1].cpu().numpy()
        v2_bg    = ~v2_infer.cell_mask
        v2_gamma[v2_bg] = 0.0
        v2_alpha[v2_bg] = 0.0

        v2_gt_path = v2_path.replace('.tif', '_gt.npz')
        if os.path.exists(v2_gt_path):
            v2_gt   = np.load(v2_gt_path)
            v2_gt_g = v2_gt["gamma"]     # blurred GT
            v2_gt_a = v2_gt["alpha"]
            cell2   = v2_infer.cell_mask

            v2_mae_g = float(np.abs(v2_gamma[cell2] - v2_gt_g[cell2]).mean())
            v2_mae_a = float(np.abs(v2_alpha[cell2] - v2_gt_a[cell2]).mean())
            print(f"  v2 Gamma MAE (cell): {v2_mae_g:.4f}  "
                  f"(v1 seen={gG_seen:.4f}, held={gG_held:.4f})")
            print(f"  v2 Alpha MAE (cell): {v2_mae_a:.4f}  "
                  f"(v1 seen={aA_seen:.4f}, held={aA_held:.4f})")

            if v2_mae_g < 2 * gG_held and v2_mae_a < 2 * aA_held:
                ov_verdict = "OK — v2 MAE within 2× of v1 held-out → no geometry memorisation"
            else:
                ov_verdict = "WARN — v2 MAE >> v1 held-out → possible spatial overfitting"
            print(f"  Verdict: {ov_verdict}")

            ov_rpt = os.path.join(RESULT_DIR, f"overfitting_test_{VERSION}.txt")
            with open(ov_rpt, "w") as f:
                f.write(f"=== Cross-video Overfitting Test [{VERSION}] ===\n")
                f.write(f"Train video : test_synthetic_cell.tif   (nested circles, hard edges)\n")
                f.write(f"Test  video : test_synthetic_cell_v2.tif (concentric rings, soft edges)\n")
                f.write(f"v1 Gamma MAE seen    : {gG_seen:.4f}\n")
                f.write(f"v1 Gamma MAE held-out: {gG_held:.4f}\n")
                f.write(f"v2 Gamma MAE (cell)  : {v2_mae_g:.4f}\n")
                f.write(f"v1 Alpha MAE seen    : {aA_seen:.4f}\n")
                f.write(f"v1 Alpha MAE held-out: {aA_held:.4f}\n")
                f.write(f"v2 Alpha MAE (cell)  : {v2_mae_a:.4f}\n")
                f.write(f"Verdict              : {ov_verdict}\n")
            print(f"  Overfitting report -> {ov_rpt}")

            # Figure: v2 pred vs v2 GT
            fig4, ax4 = plt.subplots(1, 4, figsize=(20, 4))
            im = ax4[0].imshow(v2_gamma, cmap="magma",  vmin=0, vmax=0.5)
            ax4[0].set_title(f"v2 Pred γ [{VERSION}]"); plt.colorbar(im, ax=ax4[0])
            im = ax4[1].imshow(v2_alpha, cmap="viridis", vmin=0, vmax=2.0)
            ax4[1].set_title(f"v2 Pred α [{VERSION}]"); plt.colorbar(im, ax=ax4[1])
            im = ax4[2].imshow(v2_gt_g,  cmap="magma",  vmin=0, vmax=0.5)
            ax4[2].set_title("v2 GT γ (soft)"); plt.colorbar(im, ax=ax4[2])
            im = ax4[3].imshow(v2_gt_a,  cmap="viridis", vmin=0, vmax=2.0)
            ax4[3].set_title("v2 GT α (soft)"); plt.colorbar(im, ax=ax4[3])
            fig4.suptitle(f"Cross-video Test [{VERSION}] — "
                          f"γ MAE={v2_mae_g:.3f}  α MAE={v2_mae_a:.3f}")
            fig4.tight_layout()
            ov_fig = os.path.join(RESULT_DIR, f"overfitting_test_{VERSION}.png")
            fig4.savefig(ov_fig, dpi=120, bbox_inches="tight"); plt.close(fig4)
            print(f"  Overfitting figure -> {ov_fig}")


if __name__ == "__main__":
    train_physics_reconstruction()
