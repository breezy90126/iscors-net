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
| Input | (B, K, P, P) — K=10 G_norm channels, P=64 patch |
| Encoder | 4 stages: 64→128→256→512 channels |
| Decoder | Bilinear up + skip concat, DoubleConv at each scale |
| Output | (B, 2, P, P) — γ and α |
| γ activation | Sigmoid → (0, 1) |
| α activation | ELU+1: `(F.elu(x)+1.001).clamp(max=2)` → (0.001, 2] |

The ELU+1 α activation avoids saturation at both ends (unlike Sigmoid×2).

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

v3.6 result: |Δα|=0.130, spatially separated diff map → model IS doing physics inference for α internally. TV regularization was suppressing the output expression of that structure (fixed in v3.7).

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
| **v3.7** | Separate λ_TV for γ and α | Release α boundary formation while keeping γ smooth |

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
- `inference_maps_v3.7.png` — predicted γ and α vs GT
- `loss_curve_v3.7.png` — physics loss + TV_gamma + TV_alpha curves
- `shuffle_test_v3.7.png` — τ shuffle diagnostic maps
- `generalisation_v3.7.txt` — seen vs held-out MAE report

---

## Project Structure

```
train_phys_recon.py             Main training script
datasets/phys_recon_dataset.py  G_empirical precomputation + 80/20 blind-spot
models/pissl_tau_encoder.py     U-Net, ELU+1 alpha activation
loss/phys_recon_loss.py         tau-weighted shape-only MSE
utils/traditional_iscors.py     FFT autocorrelation, G_empirical map
utils/generate_test_video.py    Synthetic 3-region video generator
EXPERIMENTS.md                  Full version history and insights
```
