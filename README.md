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

## Method (v3.x)

### Input preparation

For each pixel, compute empirical G at fixed τ lags and normalise by τ=1:

```
G_norm(τ; y,x) = G_empirical(τ; y,x) / G_empirical(τ₁; y,x)
```

This removes amplitude A; the model sees only the decay shape.

### Spatial blind-spot (self-supervision)

- 80% of cell pixels: G_norm visible in input → supervised by physics loss
- 20% of cell pixels: G_norm zeroed in input, excluded from loss
- The model must infer held-out pixels from neighbouring visible pixels

This gives a meaningful spatial generalisation test without any external GT.

### Loss: τ-weighted shape-only MSE

```
G_theory_norm(τ) = (1+γ) / (1 + γ·τ^α)

L = Σ_k w_k · (G_theory_norm(τ_k) − G_norm(τ_k))²
    w_k = log(τ_k) / Σ_j log(τ_j)
```

τ=1 gets weight 0 (no α signal); large τ up-weighted where dG/dα is strongest.

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
| γ activation | Sigmoid → (0, 1) |
| α activation | ELU+1: `(F.elu(x)+1.001).clamp(max=2)` → (0.001, 2] |

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
| **v4.0** | Masked Huber-TV (cell-cell pairs only) + α TV ×3 | Fix γ collapse: background→cell TV cascade eliminated |

---

## Running

```bash
pip install -r requirements.txt

# Generate synthetic test video (T=2000)
python utils/generate_test_video.py

# Train
python train_phys_recon.py
```

Outputs in `./result/`:
- `inference_maps_v4.0.png` — predicted γ and α vs GT
- `loss_curve_v4.0.png` — physics loss + masked TV_gamma + masked TV_alpha curves
- `shuffle_test_v4.0.png` — τ shuffle diagnostic (expect uniform diff, large |Δγ|)
- `generalisation_v4.0.txt` — seen vs held-out MAE report
- `overfitting_test_v4.0.png` — cross-video generalisation (v1 train → v2 inference)

---

## Project Structure

```
train_phys_recon.py             Main training script (v4.0: masked_huber_tv)
datasets/phys_recon_dataset.py  G_empirical precompute, 65/35 blind-spot, cell_mask_patch
models/pissl_tau_encoder.py     U-Net, ELU+1 alpha, τ positional encoding (v3.9)
loss/phys_recon_loss.py         Fisher-weighted shape-only MSE (v3.8)
utils/traditional_iscors.py     FFT autocorrelation, G_empirical map
utils/generate_test_video.py    Synthetic 3-region video generator
EXPERIMENTS.md                  Full version history and insights
```
