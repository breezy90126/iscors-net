# iSCORS-Net: Experiment Log

Physics-Informed Self-Supervised Learning for iSCAT anomalous diffusion mapping.  
Task: recover dense (γ, α) maps from a single cell video with no external labels.

---

## Physics Background

**Anomalous diffusion model (iSCORS):**

```
G(τ) = A / (1 + γ · τ^α)
```

- **γ** (gamma): diffusion speed coefficient. Higher γ → faster decay → more mobile molecule.
- **α** (alpha): anomalous exponent. α=1 normal diffusion; α<1 sub-diffusion (confined); α>1 super-diffusion (active transport).
- **A**: amplitude (proportional to particle number fluctuation; not of physical interest for dynamics).

**Synthetic test video (3 regions + background):**

| Region | γ | α | Physical interpretation |
|---|---|---|---|
| Large circle (cell body) | 0.10 | 1.0 | Normal diffusion / euchromatin |
| Small circle 1 (upper-left) | 0.50 | 1.5 | Super-diffusion / active transport |
| Small circle 2 (lower-right) | 0.05 | 0.5 | Sub-diffusion / heterochromatin |
| Background | 0.0 | 0.0 | Near-static (CV=0.05%) |

**Identifiability at τ_max=128 (T=2000):**

| Region | G_norm(128) | dG/dα(128) | Note |
|---|---|---|---|
| Cell body | 0.083 | −0.2120 | Mild decay, moderate gradient |
| Fast spot | 0.002 | −0.0000 | **Saturated — G≈0, gradient vanishes at large τ** |
| Slow spot | 0.050 | **−1.176** | Strongest gradient; α info survives at large τ |

Key counterintuitive finding: the slow spot (small γ, sub-diffusion) is the **easiest** to learn from large-τ information.
The fast spot saturates G→0 beyond τ≈16 — its α signal is only at small τ.

---

## Architecture (shared across v3.x)

**Model: `PISSLTauEncoder`** — U-Net with skip connections, bilinear upsampling.

Input: `(B, K, P, P)` — K normalised G channels, P×P spatial patch.  
Output: `(B, 2, P, P)` — γ and α maps.

**Output activations:**
- γ: Sigmoid → (0, 1)
- α: ELU+1 (Direction D) — `(F.elu(x) + 1.001).clamp(max=2.0)` → (0.001, 2.0]
  - x=0 → α=1.0 (healthy default)
  - x>0 → linear growth, no saturation for super-diffusion
  - x<0 → exponential approach to 0, non-zero gradient everywhere
  - vs. Sigmoid×2: vanishes near α≈0 and α≈2 — bad for both ends

---

## Version History

---

### v2.x — Internal Learning (Sparse GT supervision)

**Core idea:** compute traditional autocorrelation + curve-fitting at 2% of pixels; use these as GT labels to supervise a UNet on the rest.

**v2.0 (baseline):**
- FFT+Hann autocorrelation, random τ sampling, center-3×3 masked loss
- Train ratio 2%, max_tau=64, 30 epochs

**v2.1:**
- Fixed 3 root causes of training failure:
  - Background GT contamination: background pixels had high-variance noise → curve fitting produced random large γ → polluted GT. Fixed: CV < 0.5% → return (0,0).
  - Input normalization mismatch between train/inference
  - PixelShuffle → bilinear upsampling (eliminates 2×2 checkerboard artifacts)
  - γ bounded output via Sigmoid

**v2.2–v2.6:** Various hyperparameter sweeps (train_ratio, epochs, cosine LR, TV regularization, weight_decay, pretraining attempts).

**v2.7:** R² filter for real-data GT; seen vs. unseen MAE generalization report.

**v2.8–v2.9:** Huber-TV replaces L1-TV for edge-preserving smoothing.

**Insight from v2.x:** Sparse GT supervision is fundamentally limited. 2% supervised pixels → model memorizes GT locations rather than learning physics. When GT pixels are held out, MAE ratio >> 1. The network cannot generalize because the loss provides no signal at unseen pixels.

---

### v3.0 — Self-Supervised Physics Reconstruction

**Paradigm shift:** remove external GT entirely. Loss is the physics model itself.

**Core idea:** Compute empirical G(τ) = <δI(t)·δI(t+τ)>/<I²> at every pixel.  
Fit the model: minimize ||G_theory(τ; γ,α) − G_empirical(τ)||² over the whole frame simultaneously.  
The network uses spatial context (neighboring pixels) to infer parameters at held-out pixels.

**80/20 spatial blind-spot:**
- 80% of cell pixels: G_empirical visible in input → loss applied here
- 20% of cell pixels: G_empirical zeroed in input AND excluded from loss
- Forces the model to infer held-out pixels from neighbors → meaningful generalization test

**τ selection:** fixed set (1,2,4,8,16,32,48,64,96,128) — K=10 channels.

---

### v3.1 — Normalised Loss

**Problem:** amplitude A dominates the gradient (‖G_theory−G_empirical‖² is dominated by the overall scale, not the shape).

**Fix:** normalize both sides by τ=1 value:
```
G_norm(τ) = G_empirical(τ) / G_empirical(τ₁)
```
Loss becomes shape-only; A is removed from supervision entirely.

**Insight:** Without this, the model converges to predicting mean γ≈0.1 and mean α≈1.0 everywhere — the amplitude-weighted average. Shape normalization is essential to see any spatial structure.

---

### v3.2 — Amplitude-Aware (3-channel output)

**Motivation:** Try keeping amplitude as a third output channel: G_theory = A/(1+γτ^α).

**Problem:** A re-introduces the mean-regression attractor. Gradient for A is large → model learns A fast and ignores γ,α shape.

**Insight:** Predicting amplitude from G_empirical shape is fundamentally underdetermined. Separating A and normalizing (v3.1 approach) is better.

---

### v3.3 — Masked G_empirical as Input

**Change:** Use per-pixel normalised G_norm as model input (not the raw video frames).

```
g_input[y,x] = G_norm(τ₁..τ_K; y,x) for supervised pixels
g_input[y,x] = 0.0                    for held-out pixels (blind-spot)
```

**Insight:** The model now directly sees the temporal autocorrelation curve at each pixel, not the raw intensity. This aligns input representation with the physics target. The spatial blind-spot now operates in G space, not pixel space.

---

### v3.4 — Shape-Only + Masked Input (Combined)

**Combines v3.1 (shape-only loss) + v3.3 (masked G input).**

**Result:**
- γ maps improve — large circle visible
- α maps: still blurry
- Maps are grainy (pixel-level noise, no spatial smoothing)
- Lower-left of large circle poorly reconstructed

**Insight:** Without spatial regularisation, each pixel's prediction is independent. The model has no incentive to produce smooth maps even where G_empirical is consistent across neighbors.

---

### v3.5 — log-MSE + Huber-TV

**Changes:**
- log-MSE: `L = (log G_theory − log G_empirical)²` to amplify large-τ signal (where α sensitivity peaks)
- Huber-TV regularization: edge-preserving smoothing on predictions
- Extended τ to (1,2,4,8,16,32,48,64,96,128)

**Problem discovered:**
- log-MSE inflates loss scale to ~0.6 (from 7×10⁻³)
- λ_TV=0.01 → TV contributes only 0.7% of total loss → TV does nothing (TV loss rose, not fell)
- log-MSE with noisy large-τ G_empirical → unstable gradients (salt-and-pepper in log space)

**Insight:** λ scaling is not absolute — it must be proportional to the physics loss magnitude. A well-tuned λ_TV needs TV ≈ 20–50% of physics loss at early training. log-MSE changed the loss scale by 100× without adjusting λ_TV.

---

### v3.6 — τ-Weighted MSE + Scaled Huber-TV + ELU-α

**Changes (three simultaneous directions):**

1. **τ-weighted MSE** (replaces log-MSE):
   ```
   L = Σ_k w_k · (G_theory_k − G_target_k)²,   w_k = log(τ_k) / Σ log(τ_j)
   ```
   - τ=1 gets weight 0 (log(1)=0 — correct, τ=1 carries zero α signal)
   - Large τ up-weighted → α gradient amplified stably
   - Loss stays at ~10⁻² scale (same as v3.4)

2. **Scaled Huber-TV**: λ_TV=0.5 (50× larger than v3.5)
   - TV contribution ~40% of physics loss at epoch 1
   - Confirmed by TV loss declining (smoothing is active)
   - γ maps become smooth; graininess eliminated

3. **ELU+1 for α (Direction D)**:
   ```python
   alpha_map = (F.elu(p[:, 1:2]) + 1.001).clamp(max=2.0)
   ```
   - x=0 → α=1.0 default
   - x>0 → α = x+1 (linear, no saturation for super-diffusion)
   - x<0 → α = e^x (approaches 0, non-zero gradient everywhere)
   - Replaces Sigmoid×2 which saturates near both α=0 and α=2

**τ shuffle test (new diagnostic):**  
Permute the K τ-channel order in model input. Compare output maps with original.
- Large |Δγ|, |Δα| → model uses τ ordering (physics curve decoding active)
- Small |Δγ|, |Δα| → model ignores τ ordering (spatial pattern matching only)

**v3.6 shuffle results (T=2000 synthetic data):**
```
|Δgamma| (10 shuffles): 0.14 ± 0.02  → TEMPORAL
|Δalpha| (10 shuffles): 0.130 ± ...  → spatial structure clearly separated
```

**Critical finding:** The shuffle diff map for α shows the three regions (cell body / fast spot / slow spot) **clearly separated** spatially. The model IS doing physics-based temporal inference for α. But the output α map is nearly uniform.

**Diagnosis:** TV with λ=0.5 on both channels compresses α dynamic range. The model has the internal spatial representation but TV prevents it from expressing sharp region boundaries.

**Insight:** γ and α have fundamentally different physical smoothness priors:
- γ: physically smooth (diffusion coefficient varies slowly in space)
- α: can have sharp boundaries (heterochromatin vs euchromatin vs active domains)

Applying equal TV strength to both channels incorrectly suppresses α boundary formation.

---

### v3.7 — Separate TV Strengths (Current)

**Core change:** Split single `LAMBDA_TV=0.5` into channel-specific values.

```python
LAMBDA_TV_GAMMA = 0.5    # unchanged — gamma physically smooth
LAMBDA_TV_ALPHA = 0.05   # 10x weaker — release alpha boundary formation
```

Training loss:
```python
tv_g = huber_tv(preds[:, 0:1])   # TV on gamma channel only
tv_a = huber_tv(preds[:, 1:2])   # TV on alpha channel only
loss = phys_loss + LAMBDA_TV_GAMMA * tv_g + LAMBDA_TV_ALPHA * tv_a
```

**Hypothesis:** With α TV relaxed 10×, the sharp boundaries between diffusion regions that the shuffle test confirmed are encoded internally should now be expressible in the output α map.

**Expected outcome:**
- γ: remains smooth (λ=0.5 unchanged)
- α: boundaries between cell body / fast spot / slow spot become visible
- Shuffle test |Δα| may increase (model has less suppression to overcome)
- MAE on held-out α pixels should decrease

---

## Key Insights Summary

| # | Insight | Version |
|---|---|---|
| 1 | Sparse GT (2%) cannot generalize; spatial blind-spot self-supervision is needed | v3.0 |
| 2 | Amplitude A dominates gradient; shape normalization (G_norm = G/G(τ₁)) is essential | v3.1 |
| 3 | Without TV, predictions are pixel-wise independent → grainy maps despite correct loss | v3.4 |
| 4 | λ_TV must be calibrated relative to physics loss magnitude, not as an absolute value | v3.5 |
| 5 | log-MSE changes loss scale by 100×, invalidating previously tuned λ_TV | v3.5 |
| 6 | τ-weighted MSE (w=log(τ)) amplifies large-τ α signal without log instability | v3.6 |
| 7 | ELU+1 α activation: no saturation at either extreme (better than Sigmoid×2) | v3.6 |
| 8 | τ shuffle test reveals whether model uses physics ordering or spatial patterns | v3.6 |
| 9 | Shuffle test shows model HAS learned α spatial structure; TV is suppressing output | v3.6 |
| 10 | Fast spot (γ=0.5,α=1.5) saturates at large τ (G≈0); its α lives at small τ | identifiability analysis |
| 11 | Slow spot (γ=0.05,α=0.5) has STRONGEST dG/dα at large τ — counterintuitively easy | identifiability analysis |
| 12 | γ and α need different TV strengths: separate λ_TV_gamma and λ_TV_alpha | v3.7 |

---

## Open Questions

1. **v3.7 result:** Does LAMBDA_TV_ALPHA=0.05 release α boundaries while preserving γ smoothness? What is the optimal ratio?

2. **Fast spot α recovery:** G_norm(128)≈0.002 for the fast spot — essentially zero. The model must rely on τ=4–16 for its α=1.5 signal. Does τ-weighted MSE (which up-weights large τ) hurt the fast spot? Consider adding a separate loss term for small-τ regions.

3. **Real data validation:** iSCORS published results focus on chromatin condensation/relaxation. Typical γ range in high-speed iSCAT is 0.01–0.1 — the gradient-poor regime. Does the model generalize to real biological γ values?

4. **Membrane-tethered proteins:** Chromatin dynamics may approach the diffusion resolution limit at high frame rates (1000 fps). α may be dominated by confinement rather than anomalous diffusion — different physical model needed?

5. **T=2000 vs real data:** Synthetic T=2000 gives G_empirical error ≈2.3% at τ=128. Real data may have fewer frames or lower SNR — does the self-supervised signal survive?

---

## File Map

```
train_phys_recon.py           Main training script (v3.x)
datasets/phys_recon_dataset.py  G_empirical precomputation, 80/20 blind-spot split
models/pissl_tau_encoder.py   U-Net, ELU+1 alpha activation
loss/phys_recon_loss.py       tau-weighted shape-only MSE
utils/traditional_iscors.py   FFT autocorrelation, curve fitting, G_empirical map
utils/generate_test_video.py  Synthetic 3-region video (T=2000)
```
