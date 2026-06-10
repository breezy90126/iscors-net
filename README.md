# iSCORS-Net

Physics-Informed Self-Supervised Learning for iSCAT anomalous diffusion mapping.

Given a single iSCAT video (T × H × W), the network recovers dense maps of **gamma** (diffusion coefficient) and **alpha** (anomalous exponent) at every pixel — supervised only by the physics of the autocorrelation decay, with no external labels.

See [EXPERIMENTS.md](EXPERIMENTS.md) for full version history and insights.

---

## Problem

Traditional iSCORS computes autocorrelation G(τ) pixel-by-pixel and fits:

```
G(τ) = A / (1 + γ · τ^α)
```

This is O(H·W·T) and takes hours on a full cell video. We replace curve-fitting with a U-Net that learns to predict (γ, α) spatially from normalised G channels, trained self-supervisedly on the video itself.

---

## Method (v4.6)

### Input preparation — G(0)=CV² normalisation (v4.5)

For each pixel, compute empirical G at fixed τ lags and normalise by the **zero-lag**
autocorrelation G(0) = ⟨δI²⟩/⟨I⟩² = CV²:

```
G_norm(τ; y,x) = G_empirical(τ; y,x) / G(0; y,x)   →   1 / (1 + γ·τ^α)
```

G(0) is always the largest G value regardless of diffusion speed, so the ratio never
blows up (unlike the earlier `G/G(τ₁)`, which exploded when fast biological dynamics
drove G(τ=1)→0). This matches MATLAB iSCORS `nor_1` normalisation and makes **γ fully
identifiable** — the target collapses exactly to `1/(1+γτ^α)` with no τ_ref degeneracy.

> Earlier versions (v3.1–v4.3) normalised by `G(τ₁)`, giving target `(1+γτ_ref^α)/(1+γτ^α)`.
> That "shape-only" mode is retained as a fallback (`G0_NORM=False`) but G(0) is the default.

### Spatial blind-spot (self-supervision)

- 80% of cell pixels: G_norm visible in input → supervised by physics loss
- 20% of cell pixels: G_norm zeroed in input, excluded from loss
- The model must infer held-out pixels from neighbouring visible pixels

This gives a meaningful spatial generalisation test without any external GT.

### Loss: Fisher × Reliability weighted MSE (G(0) target)

```
G_theory_norm(τ) = 1 / (1 + γ·τ^α)          # matches the G(0)-normalised target

L = Σ_k w_k(τ; y,x) · (G_theory_norm(τ_k) − G_norm(τ_k))²
    w_k ∝ Fisher(τ_k) × 1/σ_G_norm(τ_k; y,x)   # normalised per pixel
```

- **Fisher τ-weighting (v3.8):** `w ∝ (∂G/∂γ)² + (∂G/∂α)²` at a prior (γ₀,α₀). Under
  G(0) normalisation all τ are informative (τ=1 is no longer zeroed — it carries γ signal).
- **Reliability weighting (v4.1):** down-weights τ channels with high measurement noise
  σ_G_norm = √((2/T)(1+G_norm²)) (Wiener–Khinchin estimate).

### TV regularisation (v3.7)

```
L_total = L_physics + λ_γ · TV(γ) + λ_α · TV(α)
```

Separate strengths: γ is physically smooth (λ_γ=0.5); α has sharp region boundaries (λ_α=0.05).

---

## Architecture

`PISSLTauEncoder` — U-Net with bilinear upsampling and skip connections.

| Component | Detail |
|---|---|
| Input | (B, 2K, P, P) — K G_norm + K τ-PE channels, P=64 patch |
| τ-PE | `tau_pe[k] = log(τ_k)/log(τ_max)` ∈ [0,1], broadcast spatially (v3.9) |
| Encoder | 4 stages: 64→128→256→512 channels |
| Decoder | Bilinear up + skip concat, DoubleConv at each scale |
| Output | (B, 2, P, P) — γ and α |
| γ activation (v4.6) | `gamma_scale · Sigmoid(x)` → (0, gamma_scale), default 2.0 |
| α activation (v4.6) | `2 · Sigmoid(x)` → (0, 2); x=0 → α=1.0 |

v4.6 replaced the v4.5 `Sigmoid→(0,1)` for γ (which clipped empirical g0_norm fits that
reach ≈1.7–2.0) and the `ELU+1.clamp(max=2)` for α (whose hard clamp piled gradients up
at α=2). Scaled sigmoids saturate smoothly at both ends with no zero-gradient pile-up.
Optionally (`use_sigma=True`) the input gains K extra σ_G_norm channels (3K total).

τ-PE makes the τ label for each G_norm channel explicit. Without it the model can use
the set of G_norm magnitudes (bag-of-values shortcut) to identify (γ,α) without learning
the τ-dependent curve shape — physics decoding only activates at region boundaries.

---

## Synthetic Test Video

Three diffusion regions + near-static background (T=2000, H=128, W=128):

| Region | γ | α | Notes |
|---|---|---|---|
| Large circle (cell body) | 0.10 | 1.0 | Normal diffusion |
| Small circle 1 (upper-left) | 0.50 | 1.5 | Super-diffusion; G saturates at large τ |
| Small circle 2 (lower-right) | 0.05 | 0.5 | Sub-diffusion; strongest dG/dα at large τ |
| Background | 0.0 | 0.0 | CV=0.05%, skipped by CV filter |

T=2000 rationale: G_empirical error ≈2.3% at τ=128 (vs 12% for T=200).

---

## τ Shuffle Test (diagnostic)

After inference, randomly permute the K τ-channel order and re-run the model. Compare output maps with original permutation.

- Large |Δγ|, |Δα| → model uses τ ordering (physics curve decoding active)
- Small |Δγ|, |Δα| → model ignores τ ordering (spatial pattern matching only)

| Version | |Δγ| | |Δα| | Pattern | Interpretation |
|---|---|---|---|---|
| v3.6 | 0.14 | 0.130 | Spatially separated regions | Physics active; TV suppressing output |
| v3.7 | 0.037 | 0.122 | Uniform (dim) | γ collapsed; log(τ) starves fast-spot gradient |
| v3.8 | 0.139 | 0.328 | **Ring (bright edge)** | Fisher weighting restores physics; interior shortcuts via spatial propagation |
| v3.9 | 0.080 | 0.393 | **Uniform** | τ-PE broke bag-of-values shortcut ✓; |Δγ| low because γ itself collapsed to ≈0 |
| v4.0 | TBD | TBD | Expected: uniform, |Δγ|↑ | Masked TV removes γ collapse → γ non-zero → larger shuffle sensitivity |

**Ring pattern (v3.8):** The shuffle diff is large at region boundaries (physics decoding needed) and small in interiors (spatial propagation from consistent neighbours suffices). This duality is expected in a U-Net; the τ-PE and increased blind-spot (v3.9) are designed to push more interior pixels into physics-decoding mode.

---

## Version Summary

| Version | Key change | Main insight |
|---|---|---|
| v2.x | Sparse GT (2%) supervision | Cannot generalize; blind-spot needed |
| v3.0 | 80/20 spatial blind-spot, physics loss | Self-supervision works |
| v3.1 | Shape-only G_norm loss | Amplitude dominates gradient; must remove |
| v3.2 | 3-ch output (γ,α,A) | A re-introduces mean-regression attractor |
| v3.3 | Masked G_norm as input | Aligns input/output in G space |
| v3.4 | v3.1 + v3.3 combined | Grainy maps without TV |
| v3.5 | log-MSE + TV | TV ineffective (λ not scaled to log-MSE magnitude) |
| v3.6 | τ-weighted MSE + scaled TV + ELU-α | TV active; shuffle test reveals internal α structure |
| v3.7 | Separate λ_TV for γ and α | Release α boundary formation; log(τ) starves fast-spot γ |
| v3.8 | Fisher τ-weighting + physics-derived TV | Fast-spot γ recovered; shuffle ring pattern diagnosed |
| v3.9 | τ-PE + 35% blind-spot | Break bag-of-values shortcut; force interior physics decoding |
| v4.0 | Masked Huber-TV (cell-cell pairs only) + α TV ×3 | Fix γ collapse: background→cell TV cascade eliminated |
| v4.1 | σ_G reliability map + Fisher×Reliability loss | Down-weight noisy τ channels per pixel |
| v4.2 | Real-data self-train + R² confidence map + checkerboard CV | Validate on real video without external GT |
| v4.5 | **G(0)=CV² normalisation** (target `1/(1+γτ^α)`) | γ fully identifiable; matches MATLAB `nor_1` |
| **v4.6** | Kurtosis artifact mask + scaled-sigmoid γ/α + Gradio frontend | Drop hot pixels; widen γ range; HF Spaces deploy |
| v4.6+ | Eval fix (γ ∝ D, not 1/D) + α variance regularizer | −0.426 was an inversion artifact; γ is actually good (checkerboard 0.80); α compression is the open problem |

> **Evaluating quality:** the trustworthy γ metric is the **checkerboard cross-validation**
> Pearson (model vs traditional curve-fit, same physics and coordinates). The `γ vs D_map`
> comparison is unit-scale-mismatched (report z-scored MAE and signed Pearson, expecting
> **positive** — γ is the decay rate, so γ ∝ D). α compression (mean-regression from
> single-power-law misspecification) is the main remaining limitation; the `LAMBDA_ALPHA_VAR`
> variance regularizer mitigates the symptom, a multi-component forward model is the cure.

---

## Running

**Synthetic trainer (diagnostics, GT available):**

```bash
pip install -r requirements.txt
python utils/generate_test_video.py     # synthetic test video (T=2000)
python train_phys_recon.py              # G0_NORM=True, GAMMA_SCALE=2.0 by default
```

Outputs in `./result/` (suffixed with the current `VERSION`, e.g. `_v4.6`):
- `inference_maps_v4.6.png` — predicted γ and α vs GT (γ colormap scales with GAMMA_SCALE)
- `loss_curve_v4.6.png` — physics loss + masked TV_gamma + masked TV_alpha curves
- `shuffle_test_v4.6.png` — τ shuffle diagnostic (expect uniform diff, large |Δγ|)
- `generalisation_v4.6.txt` — seen vs held-out MAE report
- `overfitting_test_v4.6.png` — cross-video generalisation (v1 train → v2 inference)

**Real-data pipeline & deployment:**
- `iscors_real_runner.ipynb` — Phase-2 notebook: real-video self-training, physical-unit
  conversion, R² map, checkerboard CV, model-vs-MATLAB-GT comparison.
- `app.py` — Gradio inference frontend (HF Spaces). Set `GAMMA_SCALE` / `USE_SIGMA` to
  match the checkpoint being loaded.

---

## Project Structure

```
train_phys_recon.py             Synthetic trainer (v4.6: G0_NORM, scaled-sigmoid, masked_huber_tv)
iscors_real_runner.ipynb        Phase-2 real-data notebook (v4.6 reference pipeline)
app.py                          Gradio inference frontend (HF Spaces)
datasets/phys_recon_dataset.py  G_empirical + G(0)=CV² norm, σ_G map, kurtosis artifact mask
models/pissl_tau_encoder.py     U-Net, scaled-sigmoid γ/α, τ positional encoding, optional σ input
loss/phys_recon_loss.py         Fisher × Reliability weighted MSE; g0_norm + shape_only modes
utils/traditional_iscors.py     FFT autocorrelation, G_empirical map
utils/generate_test_video.py    Synthetic 3-region video generator
EXPERIMENTS.md                  Full version history and insights
```
