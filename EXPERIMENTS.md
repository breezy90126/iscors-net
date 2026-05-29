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

### v3.7 — Separate TV Strengths

**Core change:** Split single `LAMBDA_TV=0.5` into channel-specific values.

```python
LAMBDA_TV_GAMMA = 0.5    # unchanged — gamma physically smooth
LAMBDA_TV_ALPHA = 0.05   # 10x weaker — release alpha boundary formation
```

**Actual results:**
- γ: smooth, but **severely compressed**. Fast spot pred γ≈0.15 vs GT γ=0.5.
  τ-shuffle |Δγ| collapsed to 0.037 (was 0.14 in v3.6) — model stopped using τ for γ.
- α: fast spot (α=1.5) now **visible** as a distinct yellow region. Hypothesis confirmed:
  releasing λ_TV_alpha did release the α boundaries the model had internally encoded.
- Slow spot (α=0.5): still not visible. Converges to cell body value ~0.9–1.0.
- Shuffle test: |Δγ|=0.037, |Δα|=0.122

**Root cause analysis — γ collapse:**
The log(τ) weighting concentrates gradient at τ=64–128 (the α-sensitive zone).
Fast spot has G_norm≈0 at τ>16 (saturated) → gradient is zero there.
γ signal lives at τ=2–8, which gets weight 0.025–0.063 under log(τ).
λ_TV_gamma=0.5 then overwhelms the weak γ gradient → spatial structure frozen.

**Insight:** log(τ) weighting was designed to amplify α but it systematically
starves γ at the cost of the fast spot's large-τ saturation.

---

### v3.8 — Physics-Derived TV + Fisher τ-Weighting

**Two independent fixes:**

**Direction 1 — Physics-derived TV λ:**
```
λ_TV = (σ²_G / σ²_prior / L²_PSF)   ×   correction
  σ²_G    = 1/T              (G_norm noise floor)
  L_PSF   = 0.61·λ/NA/px    (Rayleigh resolution limit)
  σ²_prior = (range/6)²     (6σ prior from activation bounds — no GT needed)
```
For T=2000, λ/NA/px = 532nm/1.4/65nm: L_PSF = 3.57 px  
→ λ_TV_gamma = 1.4e-3, λ_TV_alpha = 3.5e-4, ratio = 4× (from activation bounds)

**Direction 2 — Fisher information τ-weighting:**
```
w_k ∝ (∂G_norm/∂γ)² + (∂G_norm/∂α)²  at prior (γ₀=0.1, α₀=1.0)
```
Fisher peaks at τ=8–16 (balances both parameters); weight at τ=128 is 6.8× lower
than log(τ) → no longer over-weighting the saturated zone where fast spot lives.

**Actual results:**
- γ: fast spot now **clearly visible**, pred γ≈0.5–0.7 ✅. Cell body correct.
- α: fast spot visible. Slow spot still not recovered.
- Shuffle test: |Δγ|=0.139 (3.8× improvement), |Δα|=0.328 (2.7×) ✅
- Cross-video test (v2 concentric rings): γ MAE=0.068 (reasonable),
  **α MAE=0.458** (range 0.60–1.30; 65% relative error) ❌

**Problems identified:**
1. TV_alpha **plateau from epoch 1** — λ_TV_alpha = 3.5e-4 too small; physics noise
   floor underestimates needed strength by ~10× (initialization noise not modelled).
2. **τ shuffle ring pattern:** |diff| map shows bright ring at cell boundary,
   dim interior. Interior pixels rely on spatial propagation (not physics decoding);
   physics decoding only activates at boundaries where neighbours are inconsistent.
3. **Bag-of-values shortcut:** model can identify (γ,α) from the set of G_norm
   values without knowing τ labels (G_norm magnitude at each scale ≈ τ-independent
   fingerprint). Shuffling τ does not break this fingerprint → interior diff small.

**Insight (the core problem):** Two inference modes coexist in the U-Net:
```
Interior pixels:   spatial propagation (τ-order independent) — shortcut
Boundary pixels:   physics decoding from own τ curve (τ-order dependent)
```
The ring pattern in shuffle diff is a direct visualisation of this duality.
The modes are not bugs; they coexist by design (U-Net + spatial blind-spot).
But at 20% held-out ratio, the proportion of purely physics-decoded pixels is
too small — most interior pixels can always find consistent neighbours.

---

### v3.9 — τ Positional Encoding + 35% Blind-Spot (Current)

**Two architectural changes to reduce the spatial-propagation shortcut:**

**Direction 1 — τ positional encoding:**
```python
# Model input: (B, K, H, W) → (B, 2K, H, W)
tau_pe[k] = log(τ_k) / log(τ_max)   # (K,) ∈ [0, 1], broadcast spatially
x_in = cat([g_norm, tau_pe_spatial], dim=1)
```
The model now has **explicit τ labels** for each G_norm channel.
The bag-of-values shortcut requires identifying (γ,α) from value SET — now infeasible
because the same G_norm value at different τ labels means completely different physics.

Shuffle test semantics change: `model(full_input[:, perm, :, :])` passes shuffled
G_norm but the τ_PE registered buffer stays in correct order → mismatch at every pixel.
Expected: |Δγ|, |Δα| large and **spatially uniform** (not ring-shaped).

**Direction 2 — Increased blind-spot ratio (20% → 35%):**
```python
TRAIN_FRACTION = 0.65   # 35% held-out (was 20%)
```
At 20% held-out, a masked pixel has ~7 visible neighbours in a 3×3 window.
Consistent neighbourhood → spatial propagation is always sufficient.
At 35% held-out, isolated masked clusters appear → some pixels cannot reach
consistent neighbours → model must decode physics from own τ curve.
Microscopy spatial redundancy prior (U-Net) is preserved; per-pixel MLP is not used.

**TV correction (v3.9):**
Physics formula gives noise-floor lower bound; initialization noise adds ~10×.
```python
PHYSICS_TV_CORRECTION = 10.0
# λ_TV_gamma ≈ 1.4e-2  (vs 1.4e-3 in v3.8)
# λ_TV_alpha ≈ 3.5e-3  (vs 3.5e-4 in v3.8)
```

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
| 13 | log(τ) weighting systematically starves γ gradient: fast spot γ saturates at large τ, log(τ) concentrates weight there → γ physics loss ≈ 0 for fast spot | v3.7 |
| 14 | Fisher weighting (peaked at τ=8–16) naturally balances γ and α gradients; recovers fast spot γ | v3.8 |
| 15 | Shuffle ring pattern reveals spatial heterogeneity: interior uses spatial propagation (τ-independent), boundary uses physics decoding (τ-dependent) | v3.8 |
| 16 | Bag-of-values shortcut: G_norm magnitude patterns can identify (γ,α) without τ labels in homogeneous regions — τ ordering only essential at boundaries | v3.8 |
| 17 | τ positional encoding breaks bag-of-values shortcut: explicit τ labels make G_norm(τ_k) ↔ τ_k correspondence learnable everywhere, not just at boundaries | v3.9 |
| 18 | Increasing blind-spot ratio forces physics decoding in interior by reducing consistent-neighbour availability | v3.9 |

---

## Open Questions

1. **Slow spot α=0.5 recovery:** All versions fail here. G_norm curve for (γ=0.05, α=0.5)
   is very flat at small τ and nearly identical to (γ≈0.05, α≈1.0) shape at τ<16.
   Only τ=64–128 distinguishes them. Fisher weighting reduces large-τ weight by 6.8× vs
   log(τ). Is there a τ-weighting that preserves slow-spot α signal without hurting fast spot?

2. **TV floor calibration:** Physics formula gives correct *lower bound* but not practical
   working value. A data-driven calibration (set λ_TV to maintain TV loss ≈ 20% physics
   loss at epoch 0) would be more principled than an empirical ×10 correction.

3. **Real data validation:** iSCORS published results focus on chromatin condensation/relaxation.
   Typical γ range in high-speed iSCAT is 0.01–0.1 — the gradient-poor regime.
   Does the model generalize to real biological γ values?

4. **T=2000 vs real data:** Synthetic T=2000 gives G_empirical error ≈2.3% at τ=128.
   Real data may have fewer frames or lower SNR — does the self-supervised signal survive?

5. **Per-pixel MLP as physics-only baseline:** A shared-weight MLP over the K-dim τ curve
   (no spatial receptive field) would force pure physics decoding. Comparing its MAE to
   the U-Net would isolate the spatial denoising contribution from physics decoding quality.

---

## File Map

```
train_phys_recon.py             Main training script (v3.x)
datasets/phys_recon_dataset.py  G_empirical precomputation, 65/35 blind-spot split (v3.9)
models/pissl_tau_encoder.py     U-Net, ELU+1 alpha, τ positional encoding (v3.9)
loss/phys_recon_loss.py         Fisher-weighted shape-only MSE (v3.8)
utils/traditional_iscors.py     FFT autocorrelation, curve fitting, G_empirical map
utils/generate_test_video.py    Synthetic v1 (nested circles) + v2 (concentric rings)
```
