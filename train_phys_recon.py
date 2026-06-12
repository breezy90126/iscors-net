"""
v4.6 — G(0)=CV² normalisation + scaled-sigmoid activations (synthetic trainer).

v4.6 changes over v4.2 (sync with iscors_real_runner.ipynb / app.py pipeline):
  - G(0)=CV² normalisation anchor (v4.5): dataset target is G(τ)/G(0)=1/(1+γτ^α).
    Loss must be built with g0_norm=True so theory matches the target exactly;
    shape_only=True (legacy τ_ref normalisation) was a silent mismatch — the
    dataset already emitted G(τ)/G(0) while the loss re-normalised by G(τ_ref).
    g0_norm also selects the all-τ-informative Fisher weights (τ=1 no longer
    zeroed — under G(0) normalisation τ=1 still carries γ signal).
  - Scaled-sigmoid activations (v4.6): γ ∈ (0, GAMMA_SCALE), α ∈ (0, 2). The TV
    prior range and all γ colormaps now derive from GAMMA_SCALE, not a hard-coded
    γ_max=1 (which made λ_TV_gamma ~4× too strong and clipped γ plots at 1.0).

v4.1 post-mortem (synthetic training, real inference):
  - Synthetic v1: structure CORRECT — fast/slow/cell-body regions identified.
    γ pred [0.024, 0.362] vs GT [0.05/0.1/0.5]: dynamic range compressed at extremes.
    α pred [0.828, 1.701] vs GT [0.5/1.0/1.5]: better, but slow-spot underestimated.
    TV-α plateau immediately → alpha already smooth within regions after v4.0 fix.
  - Shuffle test: |Δγ|=0.096 |Δα|=0.155 → TEMPORAL verdict, τ-PE working.
  - Cross-video v2: γ MAE=0.070, α MAE=0.500 — γ generalises, α still off.
  - Real cell (inference only, no real training): γ mean=0.050 std=0.021,
    α mean=0.820 std=0.092. Pearson r(γ, 1/D)=−0.289 Spearman=−0.397.
    Direction correct (γ ∝ D, so γ vs 1/D is negative), magnitude weak.
    Root cause: model trained on synthetic → predicts mean, ignores spatial
    heterogeneity. G_norm mean curve fits well (theory γ=0.05 α=0.82 matches
    empirical mean), but ±1σ band very wide → heterogeneity not captured.
    Cell fraction=99.9% → CV threshold too loose for real data.

v4.2 additions:
  - real-train notebook cell: self-supervised training on real video.
    RECON_TAUS=(16,32,48,64,96,128) — excludes noisy fast-dynamics small-τ.
    Fisher prior (γ₀, α₀) auto-estimated from mean empirical G_norm curve fit.
    Combined Fisher × Reliability loss weighting inherited from v4.1.
  - R² confidence map: R²(y,x) = 1 − SS_res/SS_tot over τ.
    Quantifies per-pixel physics fit quality as posterior confidence indicator.
  - Checkerboard cross-validation: diagonal_resample(video) → gridA/gridB.
    gridA: traditional iSCORS curve-fit (quasi-GT). gridB: model inference.
    Pearson/Spearman r, MAE without external GT labels.

v4.1 additions (inherited):
  - σ_G(τ;y,x) reliability map computed in dataset (Wiener-Khinchin noise model).
  - Combined weight in loss: Fisher(τ) × 1/σ_G_norm(τ;y,x), normalised per pixel.
  - Model optionally accepts σ_G_norm as 3rd input block (3K channels, use_sigma=True).
  - Colormaps: γ and α percentile-clipped within cell mask (not full-frame).

v4.0 post-mortem:
  - Masked Huber-TV eliminated γ collapse from background→cell cascade.
  - α TV ×3 boost restored within-cell smoothing after boundary exclusion.
  - τ-PE + 35% blind-spot: TEMPORAL verdict, physics decoding confirmed active.
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

VERSION = "v4.6"

# ---- Training hyperparameters -----------------------------------------------
EPOCHS         = 200
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-4
PATCH_SIZE     = 64
TRAIN_FRACTION = 0.65                                       # v3.9: 35% blind-spot
# τ set selection — choose based on data dynamics:
#   Synthetic / slow dynamics: use full set (G(τ=1) >> 0, normalization stable)
#   Real / fast dynamics:      start at τ=16 (G(τ=1)≈0 → small-τ channels are noise)
#     Switching to fast-dynamics set requires no other code changes.
#     The loss, model τ-PE, and dataset normalization all auto-adapt to recon_taus[0].
#     Dropped channels: τ=1 (always G_norm=1, zero physics signal),
#                       τ=2,4,8 (noisy when G(τ=1)≈0 amplifies normalization error).
RECON_TAUS     = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)   # K=10  synthetic default
# RECON_TAUS  = (16, 32, 48, 64, 96, 128)                 # K=6   real / fast dynamics
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

# ---- v4.5/v4.6: normalisation anchor & activation range --------------------
# G0_NORM=True  : dataset target = G(τ)/G(0) = 1/(1+γτ^α)  (v4.5 anchor, MATLAB nor_1).
#                 Loss is built with g0_norm=True so theory matches the target and
#                 the all-τ-informative Fisher weights are used (τ=1 not zeroed).
#   G0_NORM=False: legacy τ_ref normalisation (shape_only) — kept for reproducibility.
# GAMMA_SCALE   : γ output ceiling (v4.6 scaled-sigmoid γ ∈ (0, GAMMA_SCALE)).
#                 Empirical g0_norm fits reach ≈1.7–2.0, so γ ∈ (0,1) clipped signal.
G0_NORM      = True
GAMMA_SCALE  = 2.0

# ---- v4.7: two-component forward model (toggle) ----------------------------
# N_COMPONENTS=1 (default): single power-law G_norm=1/(1+γτ^α) — UNCHANGED baseline.
# N_COMPONENTS=2: shared-α two-rate mixture
#     G_norm(τ) = f/(1+γ_fast τ^α) + (1-f)/(1+γ_slow τ^α)
#   Gives heterogeneity its own d.o.f. (f, γ_fast-γ_slow) so α represents genuine
#   anomaly instead of absorbing distribution width (fixes α mean-regression and
#   raises R² where the single power-law misfits the curve). Requires G0_NORM=True.
#   LAMBDA_OCCAM: parsimony penalty min(f,1-f)·(γ_fast-γ_slow) → collapse to a single
#   component unless the data demands two (guards the extra d.o.f. against fitting noise).
N_COMPONENTS = 1
LAMBDA_OCCAM = 0.02        # only used when N_COMPONENTS=2
# Shared-α modes (N_COMPONENTS=2 only). Precedence: FIX_ALPHA > GLOBAL_ALPHA > free.
#   FIX_ALPHA=True    : α≡1 (pure two-rate NORMAL diffusion) — diagnostic / clean ship.
#   GLOBAL_ALPHA=True : ONE learned scalar α shared over all pixels — the final ACF
#                       deliverable: a single trustworthy anomaly number, no per-pixel
#                       α blob (per-pixel α is not identifiable from one ACF).
FIX_ALPHA    = False
GLOBAL_ALPHA = False

# ---- v4.1: Reliability weighting & σ model input ---------------------------
# USE_RELIABILITY: pass σ_G_norm to loss for Fisher × Reliability combined weights.
# USE_SIGMA_INPUT: also add σ_G as 3rd input block (3K channels).
#   Set USE_SIGMA_INPUT=True only for fresh training — incompatible with v4.0 ckpts.
USE_RELIABILITY  = True
USE_SIGMA_INPUT  = False   # True enables 3K model (requires retraining from scratch)

# ---- v4.6+: α variance regularizer -----------------------------------------
# The single power-law G(τ)=1/(1+γτ^α) is misspecified for multi-component
# real curves, so the network minimises residual by collapsing α toward a mid
# value (mean-regression — the compressed α band in checkerboard CV). Hinge-
# penalise the within-cell α std below ALPHA_STD_TARGET to restore dynamic
# range; spatial coherence is supplied by the masked TV term. Set
# LAMBDA_ALPHA_VAR=0 to disable. (Kept in sync with iscors_real_runner.ipynb.)
LAMBDA_ALPHA_VAR = 0.05
ALPHA_STD_TARGET = 0.25

# ---- Fisher information τ-weighting prior -----------------------------------
# Evaluated at (γ₀, α₀) representing a "typical cell pixel" (not from GT).
# Stable training: use fixed prior, not per-pixel dynamic weights.
FISHER_GAMMA_PRIOR = 0.1   # typical cell body γ (iSCAT high-speed)
FISHER_ALPHA_PRIOR = 1.0   # baseline normal diffusion α
# -----------------------------------------------------------------------------


def compute_physics_tv_lambdas(T, wavelength_nm, na, pixel_nm, gamma_max=2.0):
    """
    Derive λ_TV for γ and α from physical quantities only.

    Formula: λ_TV = σ²_G / (σ²_prior × L²_PSF)

      σ²_G    = 1/T                  — G_norm estimation noise floor
      L_PSF   = 0.61·λ/NA/pixel     — Rayleigh limit (pixels)
      σ²_prior_γ = (γ_max/6)²       — 6-sigma prior over γ ∈ [0, γ_max]  (Sigmoid bound)
      σ²_prior_α = (α_max/6)²       — 6-sigma prior over α ∈ [0, 2]      (Sigmoid bound)

    The parameter ranges come from model activation bounds, not GT knowledge.
    γ_max tracks the model's GAMMA_SCALE (v4.6 scaled-sigmoid γ ∈ (0, GAMMA_SCALE));
    hard-coding γ_max=1 while the model emits γ ∈ (0,2) made λ_TV_gamma ~4× too strong.
    Ratio λ_γ/λ_α = (α_max/γ_max)² — physics says the wider-range parameter should
    get the proportionally weaker TV.
    """
    L_psf   = 0.61 * wavelength_nm / na / pixel_nm          # Rayleigh (pixels)
    sigma2G = 1.0 / T                                        # noise floor
    lam_g   = sigma2G / ((gamma_max / 6) ** 2 * L_psf ** 2)
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


def decode_preds(preds, n_components):
    """Map raw model output to (γ_effective, α, extras) regardless of n_components.

    1 component: preds=[γ,α]            → (γ, α, None)
    2 components: preds=[f,γ_s,γ_f,α]   → (γ_eff=f·γ_f+(1-f)·γ_s, α, {f,γ_s,γ_f})
      γ_eff is the amplitude-weighted mean rate — the single-number summary that the
      downstream maps / GT comparison / colormaps consume unchanged.
    All returned γ/α tensors keep the (B,1,H,W) channel dim.
    """
    if n_components == 1:
        return preds[:, 0:1], preds[:, 1:2], None
    f, g_slow, g_fast, alpha = (preds[:, 0:1], preds[:, 1:2],
                                preds[:, 2:3], preds[:, 3:4])
    gamma_eff = f * g_fast + (1.0 - f) * g_slow
    return gamma_eff, alpha, {"f": f, "g_slow": g_slow, "g_fast": g_fast}


def train_physics_reconstruction():
    global LAMBDA_ALPHA_VAR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{VERSION}] Using device: {device}")

    # Guard: the two-component model de-compresses α through physics; the α-variance
    # hinge would fight it and manufacture fake α structure (perinuclear blob, R²
    # collapse). They must not run together — force the hinge off.
    if N_COMPONENTS == 2 and LAMBDA_ALPHA_VAR > 0:
        print(f"[{VERSION}] [guard] N_COMPONENTS=2 → LAMBDA_ALPHA_VAR "
              f"{LAMBDA_ALPHA_VAR} → 0.0 (2-comp handles α range; hinge would fake it)")
        LAMBDA_ALPHA_VAR = 0.0

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
        T, WAVELENGTH_NM, NA, PIXEL_SIZE_NM, gamma_max=GAMMA_SCALE
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
                                predict_amplitude=False,
                                use_sigma=USE_SIGMA_INPUT,
                                gamma_scale=GAMMA_SCALE,
                                n_components=N_COMPONENTS,
                                fix_alpha=FIX_ALPHA,
                                global_alpha=GLOBAL_ALPHA).to(device)
    criterion = PhysicsReconLoss(
        recon_taus=RECON_TAUS,
        g0_norm=G0_NORM,                 # v4.5: target = G(τ)/G(0) = 1/(1+γτ^α)
        shape_only=(not G0_NORM),        # legacy τ_ref normalisation fallback
        fisher_weighted=True,
        fisher_gamma_prior=FISHER_GAMMA_PRIOR,
        fisher_alpha_prior=FISHER_ALPHA_PRIOR,
        n_components=N_COMPONENTS,
    ).to(device)
    print(f"[{VERSION}] N_COMPONENTS={N_COMPONENTS}"
          + (f"  λ_occam={LAMBDA_OCCAM}  fix_alpha={FIX_ALPHA}  global_alpha={GLOBAL_ALPHA}"
             if N_COMPONENTS == 2 else ""))
    if N_COMPONENTS == 2 and GLOBAL_ALPHA and not FIX_ALPHA:
        import torch as _t
        with _t.no_grad():
            print(f"[{VERSION}] global α (init) = "
                  f"{(2.0 * _t.sigmoid(model.alpha_global)).item():.4f}")
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
    print(f"[{VERSION}] α-variance reg: λ={LAMBDA_ALPHA_VAR}  "
          f"target α-std={ALPHA_STD_TARGET}  "
          f"({'ON' if LAMBDA_ALPHA_VAR > 0 else 'OFF'})")
    history = {"loss": [], "phys_loss": [], "tv_gamma": [], "tv_alpha": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_phys = epoch_tvg = epoch_tva = 0.0
        gamma_sum = alpha_sum = pix_count = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for g_input, g_target, train_mask, cell_mask_patch, sigma_patch in pbar:
            g_input          = g_input.to(device)
            g_target         = g_target.to(device)
            train_mask       = train_mask.to(device)
            cell_mask_patch  = cell_mask_patch.to(device)
            sigma_patch      = sigma_patch.to(device)            # (B, P, P, K)

            # sigma for model input needs (B, K, P, P)
            sigma_in = sigma_patch.permute(0, 3, 1, 2) if USE_SIGMA_INPUT else None

            optimizer.zero_grad()
            preds = model(g_input, sigma_g_norm=sigma_in)        # (B, 2, P, P)

            sigma_for_loss = sigma_patch if USE_RELIABILITY else None
            phys_loss = criterion(preds, g_target, train_mask, sigma_g_norm=sigma_for_loss)

            # Decode to effective (γ, α) so TV / variance reg / tracking are
            # identical for 1- and 2-component models (TV on γ_eff, the map we care
            # about spatially; the component split is regularised by Occam below).
            gamma_eff, alpha_ch, extras = decode_preds(preds, N_COMPONENTS)

            # Masked TV: only cell-cell adjacent pairs (excludes background-cell
            # boundaries that previously caused γ collapse via pulling cascade).
            tv_g = masked_huber_tv(gamma_eff, cell_mask_patch) \
                   if LAMBDA_TV_GAMMA > 0 else torch.tensor(0.0)
            tv_a = masked_huber_tv(alpha_ch, cell_mask_patch) \
                   if LAMBDA_TV_ALPHA > 0 else torch.tensor(0.0)
            # α variance regularizer: hinge-penalise collapsed within-cell α
            # spread (counters mean-regression from single-power-law misfit).
            cb_a = cell_mask_patch.bool()
            if LAMBDA_ALPHA_VAR > 0 and cb_a.any():
                var_pen = torch.relu(ALPHA_STD_TARGET - alpha_ch[:, 0][cb_a].std())
            else:
                var_pen = torch.zeros((), device=device)

            # Occam parsimony (2-comp only): min(f,1-f)·(γ_fast-γ_slow) is zero when
            # one component dominates (f→0/1) or the rates collapse (γ_fast→γ_slow),
            # so the model only "spends" the second component where data demands it.
            if N_COMPONENTS == 2 and LAMBDA_OCCAM > 0 and cb_a.any():
                f_c   = extras["f"][:, 0][cb_a]
                gap_c = (extras["g_fast"] - extras["g_slow"])[:, 0][cb_a]
                occam = (torch.minimum(f_c, 1.0 - f_c) * gap_c).mean()
            else:
                occam = torch.zeros((), device=device)

            loss = (phys_loss
                    + LAMBDA_TV_GAMMA * tv_g
                    + LAMBDA_TV_ALPHA * TV_ALPHA_EXTRA_SCALE * tv_a
                    + LAMBDA_ALPHA_VAR * var_pen
                    + LAMBDA_OCCAM * occam)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_phys += phys_loss.item()
            epoch_tvg  += tv_g.item() if LAMBDA_TV_GAMMA > 0 else 0.0
            epoch_tva  += tv_a.item() if LAMBDA_TV_ALPHA > 0 else 0.0

            with torch.no_grad():
                m = train_mask.unsqueeze(1)
                gamma_sum += (gamma_eff * m).sum().item()
                alpha_sum += (alpha_ch * m).sum().item()
                pix_count += m.sum().item()
                mean_g = gamma_sum / (pix_count + 1e-10)
                mean_a = alpha_sum / (pix_count + 1e-10)

            asg = alpha_ch[:, 0][cb_a].std().item() if cb_a.any() else 0.0
            pbar.set_postfix({"L": f"{loss.item():.5e}",
                              "γ": f"{mean_g:.3f}", "α": f"{mean_a:.3f}",
                              "ασ": f"{asg:.3f}"})

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
    if N_COMPONENTS == 2 and GLOBAL_ALPHA and not FIX_ALPHA:
        with torch.no_grad():
            ag = (2.0 * torch.sigmoid(model.alpha_global)).item()
        print(f"[{VERSION}] *** global α (learned) = {ag:.4f} ***  "
              f"(single cell-wide anomalous exponent; α=1 ⇒ normal diffusion)")

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
    gamma_eff_full, alpha_full, _ = decode_preds(preds_full, N_COMPONENTS)

    gamma_map = gamma_eff_full[0, 0].cpu().numpy()
    alpha_map = alpha_full[0, 0].cpu().numpy()

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
    # Percentile-clip colormaps within the cell mask (not background zeros)
    cell_m = train_dataset.cell_mask
    gm_cell = gamma_map[cell_m]; am_cell = alpha_map[cell_m]
    g_p1, g_p99 = np.percentile(gm_cell, 1), np.percentile(gm_cell, 99)
    a_p1, a_p99 = np.percentile(am_cell, 1), np.percentile(am_cell, 99)

    has_gt = gt_gamma is not None
    ncols  = 4 if has_gt else 2
    fig2, axes2 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    im0 = axes2[0].imshow(gamma_map, cmap="magma",  vmin=g_p1, vmax=g_p99)
    axes2[0].set_title(f"Pred Gamma [{VERSION}]  [{g_p1:.3f},{g_p99:.3f}]")
    plt.colorbar(im0, ax=axes2[0])
    im1 = axes2[1].imshow(alpha_map, cmap="viridis", vmin=a_p1, vmax=a_p99)
    axes2[1].set_title(f"Pred Alpha [{VERSION}]  [{a_p1:.3f},{a_p99:.3f}]")
    plt.colorbar(im1, ax=axes2[1])
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
            gs_eff, as_full, _ = decode_preds(preds_shuf, N_COMPONENTS)
            dg = (gamma_eff_full[0, 0] - gs_eff[0, 0]).abs()
            da = (alpha_full[0, 0] - as_full[0, 0]).abs()
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
    gv_eff, av_full, _ = decode_preds(preds_vis, N_COMPONENTS)
    gm_shuf = gv_eff[0, 0].cpu().numpy(); gm_shuf[bg_mask] = 0.0
    am_shuf = av_full[0, 0].cpu().numpy(); am_shuf[bg_mask] = 0.0
    diff_g  = np.abs(gamma_map - gm_shuf)
    diff_a  = np.abs(alpha_map - am_shuf)

    fig3, ax3 = plt.subplots(2, 3, figsize=(15, 8))
    for row, (orig, shuf, diff, cmap, vmax, lbl) in enumerate([
        (gamma_map, gm_shuf, diff_g, "magma",   GAMMA_SCALE, "Gamma"),
        (alpha_map, am_shuf, diff_a, "viridis", 2.0,         "Alpha"),
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
        v2_g_eff, v2_a_full, _ = decode_preds(v2_preds, N_COMPONENTS)

        v2_gamma = v2_g_eff[0, 0].cpu().numpy()
        v2_alpha = v2_a_full[0, 0].cpu().numpy()
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
