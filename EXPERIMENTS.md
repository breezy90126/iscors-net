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

### v3.9 — τ Positional Encoding + 35% Blind-Spot

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

**Actual results:**
```
Gamma MAE seen    : 0.039   held-out: 0.036  ratio: 0.92
Alpha MAE seen    : 0.190   held-out: 0.175  ratio: 0.92
v2 Gamma MAE      : 0.061   (OK)
v2 Alpha MAE      : 0.501   (WARN — slightly worse than v3.8's 0.458)
|Δγ| shuffle      : 0.080   (down from 0.139 — see diagnosis below)
|Δα| shuffle      : 0.393   (up from 0.328 — highest so far)
Shuffle pattern   : uniform (v3.8 ring → v3.9 uniform) ✅
```

**What worked:**
- α map shows all three regions clearly for the first time — fast spot (yellow), cell body
  (teal), slow spot (dark blue). τ-PE effect confirmed: bag-of-values shortcut broken.
- held-out MAE < seen MAE: 35% blind-spot forces stronger spatial generalisation. Held-out
  pixels benefit from spatially averaged predictions from well-trained neighbours.
- |Δα| shuffle = 0.393 (highest yet): α physics decoding maximally active.

**What failed:**
1. **γ cell body and slow spot still near-zero.** Only fast spot (γ=0.5) visible.
2. **TV_alpha plateau at 0.1 throughout training.** λ_TV_alpha = 3.54e-3 appears not
   to decrease the unscaled TV value.
3. **|Δγ| shuffle decreased 0.139 → 0.080** — not τ-PE failure. γ is near-zero in the
   cell body → shuffling τ doesn't change a near-zero prediction → diff artificially small.

**Root cause of γ collapse (diagnosed for v4.0):**
`huber_tv` was applied to the full patch (B,1,P,P) with no cell mask.
Background pixels receive G_norm=0 input → model learns γ≈0 for them (no physics loss).
Huber-TV across background-cell boundaries penalises `|γ_cell_edge − γ_background|`,
pulling cell-edge γ toward 0. This cascade propagates inward through the TV chain:
```
background (γ≈0) ← TV → cell edge ← TV → cell interior → γ collapse
```
Cell body (γ=0.1) and slow spot (γ=0.05) are small enough to be fully pulled to zero.
Fast spot (γ=0.5) survives because the physics gradient is large enough to resist.

**TV_alpha plateau root cause (same mechanism):**
Background-cell boundary differences are large (γ=0 vs γ=0.1) and dominate the TV
mean value, creating a constant floor. Within-region TV is a small fraction of the total.

---

### v4.0 — Masked Huber-TV (Current)

**Fix: Apply TV only between cell-cell adjacent pixel pairs.**

```python
def masked_huber_tv(x, cell_mask, delta=0.05):
    m  = cell_mask.unsqueeze(1).float()
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    mx = m[..., 1:] * m[..., :-1]      # 1 only where both pixels are cell
    my = m[..., 1:, :] * m[..., :-1, :]
    n  = mx.sum() + my.sum() + 1e-10
    return (_h(dx) * mx).sum() / n + (_h(dy) * my).sum() / n
```

`cell_mask_patch` is added as a 4th return element from `PhysReconDataset.__getitem__`.

**Why this fixes both problems:**
- γ collapse: background-cell boundary TV penalty eliminated → cell-edge γ no longer
  pulled toward background → physics loss can drive γ toward true values.
- TV_alpha plateau: background-cell boundary floor removed → within-cell TV_alpha
  becomes the dominant term → effective smoothing is now visible.

**α TV boost (Fix 2):**
```python
TV_ALPHA_EXTRA_SCALE = 3.0
# effective λ_TV_alpha = 3.54e-3 × 3.0 = 1.06e-2
```
After masking, fewer pairs contribute (cell-only), so absolute TV magnitude decreases.
×3 compensates and ensures TV_alpha ≈ 10% of physics loss — the target operating range.

**Actual results:**
```
|Δγ| shuffle      : 0.199  (v3.9: 0.080 → +2.5×)  ← most important number
|Δα| shuffle      : 0.299  (v3.9: 0.393 → slightly lower)
v2 Gamma MAE      : 0.077  (v3.9: 0.061 → slight regression)
v2 Alpha MAE      : 0.510  (v3.9: 0.501 → similar)
TV_gamma unscaled : ~5e-3 floor (same as v3.9 — working as before)
TV_alpha unscaled : 0.25 → 0.025  (v3.9: stuck at 0.1 → Fix 2 CONFIRMED)
```

**Fix 1 — γ recovery assessment:**
Visually the γ map looks similar to v3.9 (fast spot bright, cell body dark). This is
**not necessarily failure** — GT γ=0.1 on a [0,1] colormap IS visually near-black,
identical to γ=0. The reliable diagnostic is the shuffle test:

| Metric | v3.9 | v4.0 | Implication |
|---|---|---|---|
| |Δγ| cell mean | 0.080 | **0.199** | γ is now used in physics decoding throughout cell |
| γ map visual | fast spot only | fast spot only | Cell body γ=0.1 just looks dark at [0,1] scale |

The |Δγ| 2.5× increase means cell body γ is **non-trivially nonzero** and changes
when τ is shuffled. Confirmed: masked TV partially or fully restored γ prediction.
Without the generalisation MAE numbers (not in zip), exact cell body γ accuracy is
unknown — but the shuffle sensitivity increase is the primary evidence for recovery.

**Fix 2 — TV_alpha confirmed:**
v3.9: TV_alpha constant at 0.1 (boundary terms dominated → no smoothing)
v4.0: TV_alpha 0.25 → **0.025** (4× lower floor). Background-cell boundary pairs
removed → within-cell smoothing is the only remaining TV signal → it decreases as
the α map converges. Fix 2 root cause diagnosis confirmed correct.

**|Δα| decreased 0.393 → 0.299:**
Expected side-effect. Smoother α map (TV now active) → smaller per-pixel differences
when τ shuffled. This is NOT a regression — it reflects α being more spatially
regularized (less noisy), trading some shuffle sensitivity for spatial consistency.

**Cross-video test (v2):**
γ MAE 0.061 → 0.077, α MAE 0.501 → 0.510. Slight regression. Masked TV changed
the training dynamics (stronger within-cell α smoothing) without changing the
fundamental spatial overfitting to v1 geometry. This remains an open problem.

**Real data (COBRI_rarw_video.tif, T=2000, binned 320×320):**
```
Cell pixels : 99.9%  (CV filter not separating BG — full-FOV cell video)
gamma p1–p99: [0.017, 0.117]   mean=0.051  std=0.021
alpha p1–p99: [0.619, 1.061]   mean=0.812  std=0.095
```
Gamma range [0.017–0.117] and alpha [0.62–1.06] are physically consistent with
chromatin dynamics (sub-normal to normal diffusion). 99.9% cell pixels likely
correct for a full-FOV cell video where background is outside the frame.
Traditional iSCORS MAT fields are BG_img/Cond_map/D_map/V_map (not gamma/alpha); notebook
updated to extract by exact names and compare model gamma vs 1/D_map (inverse relationship).

---

### v4.1 — σ_G Reliability Weighting

**Add a per-pixel, per-τ measurement-noise map and use it to weight the loss.**

```
σ²_G(τ) ≈ (2/T)·[G(0)² + G(τ)²]        # Wiener–Khinchin noise estimate
σ_G_norm = σ_G / G(τ_ref)               # in the normalised G space
w_combined(τ;y,x) = Fisher(τ) × 1/(σ_G_norm(τ;y,x)+ε)   # normalised per pixel
```

Fisher says *which τ carries physics information*; reliability says *which τ is
trustworthy at this pixel*. The product down-weights noise-dominated channels
(small-τ for fast real dynamics) without dropping them globally. Optionally σ_G_norm
is also fed as a 3rd input block (`use_sigma=True`, 3K channels) so the network can
learn the down-weighting itself — requires training from scratch (incompatible with
2K checkpoints). Backward compatible: σ=None falls back to Fisher-only weighting.

---

### v4.2 — Real-Data Pipeline + Confidence + Cross-Validation

Three additions for working on real video with **no external GT**:

1. **Real self-training cell** (`iscors_real_runner.ipynb`): train directly on a real
   iSCAT video. `RECON_TAUS=(16,32,48,64,96,128)` drops noise-only small-τ; Fisher prior
   (γ₀,α₀) auto-estimated from the mean empirical G_norm curve fit.
2. **R² confidence map**: `R²(y,x) = 1 − SS_res/SS_tot` over τ — per-pixel physics-fit
   quality as a posterior confidence indicator.
3. **Checkerboard cross-validation**: diagonal resample → gridA / gridB. gridA gets a
   traditional curve-fit (quasi-GT), gridB gets model inference; report Pearson/Spearman
   and MAE with no external labels.

**Real-data post-mortem:** structure direction correct but heterogeneity weak —
`Pearson r(γ, 1/D_map) ≈ −0.29`. A synthetic-trained model regresses to the mean;
the mean G_norm curve fits well but the ±1σ band is wide → spatial heterogeneity
under-captured. Motivated training directly on real video (v4.2 real cell).

---

### v4.5 — G(0)=CV² Normalisation (key correctness change)

**Change the normalisation anchor from G(τ_ref) to the zero-lag G(0).**

```
G(0;y,x) = ⟨δI(t)²⟩_t / ⟨I⟩²  = CV²        # always the largest G value
G_norm(τ) = G(τ)/G(0) = 1 / (1 + γ·τ^α)    # exact, no τ_ref degeneracy
```

**Why it matters:**
- `G/G(τ₁)` blows up for fast biological dynamics where G(τ=1)→0 (insight #26).
  G(0) is always the maximum, so the ratio is bounded for any diffusion speed.
- The target collapses to `1/(1+γτ^α)` → **γ is fully identifiable** (the old
  `(1+γτ_ref^α)/(1+γτ^α)` form had a τ_ref-dependent scale degeneracy with α).
- Matches MATLAB iSCORS `nor_1` (CorrF/CorrF(1)) → model γ and MAT `1/D_map` are
  directly comparable on the same scale.

**Loss/Fisher follow the anchor:** with `g0_norm=True` the theory is `1/(1+γτ^α)` (no
self-normalisation) and the Fisher weights switch to `_fisher_weights_g0` — under G(0)
normalisation τ=1 is still γ-informative, so it is no longer zeroed (it was under the
τ_ref formula). σ_clip becomes a safe no-op since σ_G_norm_g0 ≤ √(4/T) ≪ 2.

---

### v4.6 — Artifact Mask + Scaled-Sigmoid + Gradio Frontend (Current)

**Three changes:**

1. **Kurtosis artifact mask** (`phys_recon_dataset.py`): camera saturation, hot/dead
   pixels and debris produce spiky temporal traces (a few extreme frames dominate the
   variance) with excess kurtosis far above the rest of the cell. These masquerade as
   extreme-γ outliers (dark/blown-out blobs) because their G(τ) curves don't follow
   diffusion physics. Flag `kurtosis > max(2×p99_cell, 10)` and drop from `cell_mask`
   before normalisation.

2. **Scaled-sigmoid activations (Direction E):**
   ```
   γ = gamma_scale · sigmoid(x)   → (0, gamma_scale),  default 2.0
   α = 2 · sigmoid(x)             → (0, 2),  x=0 → α=1.0
   ```
   - γ: empirical g0_norm fits (checkerboard gridA, 1/D_map GT) reach ≈1.7–2.0;
     the v4.5 plain `Sigmoid→(0,1)` was clipping real signal at the top. `gamma_scale=2.0`
     gives headroom while staying stable (unbounded Softplus risks early blow-up).
   - α: `2·sigmoid` is symmetric and saturates smoothly at both ends — no zero-gradient
     boundary "pile-up" like the v4.5 hard `.clamp(max=2.0)` produced at α=2.

3. **Gradio inference frontend (`app.py`):** HF Spaces deployment — upload video →
   background removal → `PhysReconDataset` (reuses the G(0) normalisation) → checkpoint
   inference → (γ,α) maps, physical-unit conversion, and optional MATLAB-GT comparison.
   `GAMMA_SCALE`/`USE_SIGMA` must match the loaded checkpoint.

**TV prior follows γ range:** `compute_physics_tv_lambdas` now takes `gamma_max=GAMMA_SCALE`;
hard-coding γ_max=1 while the model emits γ∈(0,2) made λ_TV_gamma ~4× too strong. The
synthetic trainer `train_phys_recon.py` was synced to the notebook here: `g0_norm=True`,
`gamma_scale` passed through, γ colormaps scale with GAMMA_SCALE.

---

### v4.6+ — Baseline post-mortem: evaluation fix + α variance regularizer

After re-training a clean v4.6 baseline on the real cell, the headline `γ vs 1/D_map`
Pearson read **−0.426** — looking like failure. Diagnosis: that number was an
**evaluation artifact**, not a model failure.

**Evaluation fix (γ ∝ D):** γ is the decay RATE in G(τ)=1/(1+γτ^α), so faster
diffusion (larger D) → larger γ → **γ ∝ D, not 1/D**. The earlier `D → 1/D`
inversion flipped the sign. Comparing γ vs `D_map` directly gives the expected
positive correlation, and the raw MAE (1.58) was meaningless because γ∈(0,2) and D
live on different unit scales — replaced by a **z-scored MAE**. The trustworthy
number was always the **checkerboard CV** (model gridB vs traditional curve-fit
gridA, same physics + coords, no unit gap): **γ Pearson 0.80**, confirming γ is well
recovered. Applied to `iscors_real_runner.ipynb` (p2-gt-compare) and `app.py`.

**α compression is the real problem.** Checkerboard α Pearson 0.64 but the dynamic
range collapses to ~0.5–0.9 (traditional fit spans 0.2–2.0). Root cause is a
loss-landscape loophole: the single power-law is misspecified for multi-component
real curves (R²≈0.70, theory misses the mean G_norm curve at mid-τ), so the minimum-
MSE strategy is to predict a mid α everywhere (mean-regression); α's gradient is also
shallow (signal only at noisy, reliability-down-weighted large τ).

**α variance regularizer (band-aid):** hinge-penalise within-cell α std below
`ALPHA_STD_TARGET` (`LAMBDA_ALPHA_VAR·relu(target − std(α_cell))`) so "predict the
mean" is no longer free; spatial coherence comes from the existing adaptive TV.
Tunable, off by setting λ=0; live α-std shown in the progress bar. This treats the
symptom — the cure (single-power-law misspecification, α/σ_D degeneracy) needs a
multi-component forward model or a second projection (STICS/DDM, see brief).

---

### v4.7 — Two-component forward model (toggle: `N_COMPONENTS=1/2`)

Addresses the α-compression root cause (not the symptom). The single power-law forces
α to absorb BOTH genuine sub-diffusion AND heterogeneity-induced stretching — they are
confounded in one d.o.f., so α regresses to a mid value and R²≈0.70.

**Model (`N_COMPONENTS=2`):** shared-α two-rate mixture
```
G_norm(τ) = f / (1 + γ_fast·τ^α) + (1-f) / (1 + γ_slow·τ^α)
```
Output `[f, γ_slow, γ_fast, α]` with `γ_fast = γ_slow + softplus(Δ)` (ordering breaks the
label-swap symmetry). Heterogeneity now lives in `(f, γ_fast-γ_slow)`, freeing α to mean
genuine anomaly → α should de-compress and R² should rise where the curve is "fatter"
than a single Lorentzian. Post-hoc `μ_D ∝ f·γ_fast+(1-f)·γ_slow` and a σ_D-like spread
come for free, without the Laplace-inversion integral.

**Identifiability safeguards (essential — 4 params from 10 noisy τ is more ill-posed):**
- ordering `γ_fast ≥ γ_slow` (softplus Δ) removes the two-equivalent-minima degeneracy;
- Occam penalty `LAMBDA_OCCAM · min(f,1-f)·(γ_fast-γ_slow)` collapses to one component
  unless the data demands two (it is a strict superset of single-component, so it can
  never do worse) — guards the extra d.o.f. against fitting noise;
- existing adaptive TV supplies spatial coherence.

Default is `N_COMPONENTS=1` (the v4.6 baseline, byte-identical). Toggle to 2 and compare
**R²** (should rise from ~0.70) and **within-cell α std** (should rise without the α-variance
band-aid) before adopting. Wired through `train_phys_recon.py`, `iscors_real_runner.ipynb`
(train / inference / R² / checkerboard), and `app.py`.

---

### v4.7 — ACF line conclusion: global-α deliverable + honest scope

The α=1 diagnostic (`FIX_ALPHA`, VERSION `-a1`) and the GPU classical baseline resolved
what a single ACF can and cannot deliver. The decisive number is the **R² ladder**:

| Model | R² (model / classical per-pixel) |
|---|---|
| single-component (γ,α) | ~0.70 |
| 2-component, free α | 0.91 / 0.96 |
| 2-component, α=1 (FIX_ALPHA) | 0.886 / 0.91 |

- **The big jump (0.70→0.91) is heterogeneity** (the two rates f, γ_fast, γ_slow), not α.
- **Forcing α=1 costs only ~0.03–0.05 R²** → the apparent anomaly is ~95% heterogeneity
  with a small genuine anomalous component. The perinuclear/edge **blob vanished** with
  α=1, and γ got *cleaner* (model-vs-classical Pearson 0.853→0.890; checkerboard γ 0.874,
  MAE 0.016) — confirming the free per-pixel α was a weakly-identified nuisance.

**Why R² is not α-confidence (the blob proof):** the blob sits on HIGH R² (~0.9) while its
α is arbitrary. R² is whole-curve fit quality; it is invariant to α once (f, γ_fast,
γ_slow) explain the curve. The parameter-specific confidence is the Fisher curvature
`I_αα = Σ_τ (∂G/∂α)²/σ_G²`, Cramér–Rao `σ_α ≈ 1/√I_αα` (the `p2-alpha-identifiability`
cell). A per-pixel α that is BOTH unconfounded AND identifiable is **not extractable from
one ACF** — it needs a second projection (STICS).

**Final ACF deliverable (`GLOBAL_ALPHA`, VERSION `-ga`):**
- **γ map** — the trustworthy primary product (checkerboard 0.80–0.87, classical 0.85–0.89).
- **Heterogeneity maps** f, γ_fast, γ_slow — what actually carries the curve shape.
- **ONE global α scalar** — a single cell-wide anomalous exponent, pinned jointly by every
  pixel (no per-pixel α freedom → no blob). Recovers the small real anomaly as one number.
- **α-identifiability map** — honest degradation: where α could be measured if free.
- **NOT** a per-pixel α map (a single ACF cannot identify it).

**Three shared-α modes (N_COMPONENTS=2), precedence FIX_ALPHA > GLOBAL_ALPHA > free:**
free per-pixel α (blob; diagnostic only) · `FIX_ALPHA` α≡1 · `GLOBAL_ALPHA` one scalar (ship).

**Speed (ablation, GPU):** `utils/gpu_iscors_fit.py` fits dense (γ,α) for all cell pixels
in seconds — the same scale as one ML forward pass. The README's "hours" was a CPU/scipy
per-pixel artifact, not a computational barrier. So the ML never won on *speed*; its only
remaining justification is *quality* (spatial denoising), which the classical baseline now
measures directly. The checkerboard quasi-GT was upgraded to this GPU fitter (same model
family) so the cross-validation is apples-to-apples and fast.

**Next line — STICS.** A per-pixel α needs the independent spatial projection: STICS encodes
α (τ-scaling of the spatial spread) and σ_D (spatial non-Gaussianity at fixed τ) as separable
features, breaking the α/σ_D degeneracy — but only where the spread exceeds the PSF (honest
degradation again). ACF and STICS then cross-validate (γ/heterogeneity ↔ σ_D/v).

---

### v4.7 final (v4.6-1ga) — ACF-line conclusion: classical wins, α is unverifiable

The single-component global-α run + the GPU iMSD verifier closed the line.

**iMSD α verifier: NOT RESOLVED (and that is itself the answer).** σ²_space(τ) does not grow
with τ — it hovers at the PSF floor (~11 px²), so the fit hits the α=0.1 bound (an artifact,
not a measurement). The diffusion spread stays below the PSF over τ=1–128. Meaning:
- STICS/iMSD **cannot arbitrate α** for this data at this resolution (slow/confined dynamics —
  the brief's first-to-fail degradation). Per-region/sliding-window iMSD would be worse, not better.
- But it revealed real physics: the ACF decorrelates strongly in time while STICS shows **no
  spatial spreading** → the dynamics are **in-place / confined, not translational diffusion**.
  So the recovered α is a *temporal-decorrelation* exponent, not a verified transport exponent.

**α converged across methods at ≈0.6** once the spec is clean (single-component, global α):
classical B=0.61, classical E=0.69, **1-comp global-α U-Net=0.63**. The earlier 0.85 was a
**2-component shared-α artifact**; in the correct spec the U-Net and classical AGREE. So the
"ML α biased high" worry was a model-spec artifact, not an ML property.

**Speed: GPU classical wins decisively for the actual use (per-video self-supervised).**
GPU-batched classical fit = seconds, no training; the U-Net needs minutes–hours of per-video
training. The ML only wins at high throughput with a pre-trained model (ms inference), which is
not how this is used. So ML is not justified on speed.

**ML value reduces to one question (head-to-head):** does the U-Net's γ beat its classical twin
(same spec, B) in held-out self-consistency? `p2-gamma-headtohead` reports `pM − p_matched` and
writes `headtohead_VERSION.txt` with an explicit KEEP/RETIRE verdict. Even a positive gain cannot
be proven to be denoising rather than smoothing (no true GT; STICS can't arbitrate). Given equal
α, high γ agreement, and the classical speed win, the evidence already favours **retiring the
U-Net for the γ/α deliverable**.

**ACF-line deliverable (honest):**
- **γ map** — primary, method-consistent; produced fastest by the GPU classical 1-component fit.
- **apparent global α ≈ 0.6** — consistent across classical and ML; labelled a *temporal-
  decorrelation* exponent (real-vs-heterogeneity/confinement **not** verifiable from this data).
- **iMSD** — reports "sub-PSF, unresolved" (honest degradation) and shows the motion is in-place.
- **Open / next:** generalisation tests (rotation / augmentation transforms) probe whether the
  U-Net memorises geometry — relevant only if the network is kept; the classical fit has no
  geometry-memorisation to test. Verifying α would need higher spatial resolution / faster
  dynamics / a different lever, not more model complexity.

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
| 19 | TV applied across background-cell boundaries causes γ collapse: background γ≈0 (no physics loss) + TV chain → cell-edge γ → 0 → cascades inward | v3.9 diagnosis |
| 20 | Masked Huber-TV (cell-cell pairs only) breaks the background→cell TV cascade; each fix (γ collapse, TV plateau) has the same root cause and same solution | v4.0 |
| 21 | γ=0.1 on a [0,1] colormap is visually near-black (same as γ=0). Visual inspection is unreliable for small γ recovery; |Δγ| shuffle is the correct diagnostic | v4.0 |
| 22 | TV_alpha plateau was entirely from background-cell boundary terms: masked TV reduced unscaled TV_alpha floor 0.1 → 0.025 (4×), confirming the root cause | v4.0 |
| 23 | Stronger α TV regularisation (×3) reduces |Δα| shuffle (0.393 → 0.299): smoother maps → smaller per-pixel shuffle diff. Not a regression — expected trade-off | v4.0 |
| 24 | Real data gamma [0.017–0.117] and alpha [0.62–1.06] match expected chromatin dynamics. 99.9% cell pixels is correct for full-FOV microscopy (no background margin) | v4.0 |
| 25 | τ shuffle spatial heterogeneity persists even with τ-PE: cell body has highest |Δγ| (γ=0.1, G decays moderately — physics identifiable); fast spot has highest |Δα| (α lives at small τ, hard to shuffle away); slow spot has lowest both (G_norm near-flat at large τ — physics barely distinguishable from noise). τ-PE cannot force identifiability where the signal itself is weak. | v4.0 real |
| 26 | G_norm normalization is unstable for real data at small τ: biological dynamics are fast → G(τ=1)≈0 → G_norm=G(τ)/G(τ=1) amplifies noise → τ=1,2,4,8 channels are effectively pure salt-and-pepper noise. 4 of 10 model input channels carry no physics signal. Use τ_min≥16 for real data, or normalize by G(τ_min_nonzero) instead of G(τ=1). | v4.0 real |
| 27 | Traditional iSCORS MAT output fields are BG_img (cell morphology), Cond_map (V_DLS/D condensation), D_map (diffusion coefficient), V_map (velocity). Our model's γ is proportional to 1/D_map (inverse: high γ → fast decay → small D). α has no traditional equivalent. MAT extraction must use exact field names; generic search ('gamma','alpha') finds nothing. | v4.0 real |
| 28 | Noisy small-τ channels can be excluded by changing RECON_TAUS=(16,32,48,64,96,128). No other code changes needed — dataset, model τ-PE, and loss all auto-adapt. The loss Fisher weights must use τ_ref=recon_taus[0] in their partial derivative formulas; the legacy τ_ref=1 formula was a hidden bug for any τ set not starting at 1. | v4.1 candidate |
| 29 | Fisher (which τ has info) × Reliability 1/σ_G (which τ is trustworthy) is the right combined weight: it down-weights noisy channels per pixel without dropping them globally. σ_G can also be a 3rd input block so the network learns its own down-weighting. | v4.1 |
| 30 | G(0)=CV² is the only normalisation anchor that is bounded for all diffusion speeds (always the max G). Normalising by G(τ₁) blows up when G(τ=1)→0 for fast dynamics. G(0) also removes the τ_ref scale degeneracy → γ fully identifiable, and matches MATLAB nor_1 so model γ ↔ 1/D_map on one scale. | v4.5 |
| 31 | Under G(0) normalisation τ=1 still carries γ signal, so it must NOT be zeroed — Fisher weights switch from the τ_ref formula (which zeroes τ_ref) to the g0 formula. Loss theory must drop self-normalisation (g0_norm=True); leaving shape_only=True silently mismatches the dataset target. | v4.5 |
| 32 | Saturated/hot pixels have abnormally high temporal-trace kurtosis and produce non-physical G(τ) curves → extreme-γ blobs. Excess-kurtosis masking before normalisation removes them more reliably than a CV threshold alone. | v4.6 |
| 33 | Activation range must match the data: empirical g0_norm fits reach γ≈1.7–2.0, so γ∈(0,1) clips signal — use scaled-sigmoid γ∈(0,gamma_scale). Hard clamps (ELU+1.clamp at α=2) pile gradients up at the boundary; smooth saturating sigmoids avoid this. TV priors and γ colormaps must track gamma_scale, not a hard-coded γ_max=1. | v4.6 |
| 34 | γ ∝ D (γ is the decay rate; faster diffusion → larger γ). Comparing γ vs 1/D_map gave a spurious −0.426 Pearson that masqueraded as model failure. Compare vs D_map directly; the real metric is checkerboard-CV γ (0.80). Raw MAE across γ/D unit scales is meaningless — z-score first. | v4.6+ |
| 35 | α mean-regression is a loss-landscape loophole, not an evaluation issue: single-power-law misfit makes a mid-α the minimum-MSE answer, and α's gradient is shallow (large-τ only, down-weighted by reliability). A within-cell α-variance hinge removes the loophole, but the cure is a multi-component forward model / second projection — α and σ_D are degenerate under a single ACF. | v4.6+ |
| 36 | The R² ladder separates cause: single→2-component is +0.21 R² (heterogeneity, real, large); 2-comp free-α→α=1 is only −0.03–0.05 R² (the genuine anomaly, small). The apparent sub-diffusion was ~95% heterogeneity. Forcing α=1 also removed the blob AND improved γ consistency — the free per-pixel α was destabilising the whole fit. | v4.7 |
| 37 | R² ≠ parameter confidence: the blob has high R² but arbitrary α. R² is whole-curve fit quality, invariant to α once f/γ explain the curve. Use Fisher curvature σ_α≈1/√(Σ(∂G/∂α)²/σ_G²) for honest per-parameter confidence. | v4.7 |
| 38 | A per-pixel α that is both unconfounded and identifiable is not extractable from one ACF — information limit, not a modelling failure. Ship γ + heterogeneity (f,γ_fast,γ_slow) + ONE global α scalar (jointly pinned, no blob); push per-pixel α to STICS. | v4.7 |
| 39 | GPU-batched classical fitting is seconds (≈ one ML forward pass); the "hours" was a CPU/scipy per-pixel artifact. ML acceleration of ACF is therefore a quality claim, not a speed claim — and the classical baseline measures the quality delta directly. | v4.7 |

---

## Open Questions

1. **Quantitative γ recovery (cell body, slow spot):** v4.0 |Δγ| shuffle 0.199 confirms
   γ is non-trivially nonzero. But exact MAE for cell body (GT=0.1) and slow spot (GT=0.05)
   is unknown (generalisation txt not in zip). Needed: run with full output capture and
   compare γ MAE seen/held-out to confirm cell body is recovered, not just fast spot.

2. **Colourmap-aware inference plot:** Current plot uses vmax=1.0, making γ=0.1 visually
   dark. Add a second plot with vmax=0.2 to reveal cell body structure. This was the
   source of the "γ collapse" misdiagnosis — visual inspection fooled by scale.

3. **Cross-video spatial overfitting:** v2 MAE unchanged (γ 0.077, α 0.510) across all
   versions. The model memorizes v1 geometry regardless of TV masking. Fundamental
   cause: the physics loss itself provides geometry-specific G_empirical patterns.
   Possible fix: augment with random spatial crops, flips, or multi-video training.

4. **Real data τ selection:** τ=1,2,4,8 are noise-only for fast biological dynamics.
   Fix: set `RECON_TAUS=(16,32,48,64,96,128)` in train_phys_recon.py. Dataset normalises
   by G(τ=16); loss computes G_theory_norm=(1+γ·16^α)/(1+γτ^α); Fisher weights use τ_ref=16.
   All auto-adapt to recon_taus[0] after the Fisher weight fix in loss/phys_recon_loss.py.
   Also measure Pearson r(model gamma, 1/D_map) to validate real-data accuracy.

5. **Per-pixel MLP as physics-only baseline:** A shared-weight MLP over the K-dim τ curve
   (no spatial receptive field) would force pure physics decoding. Comparing its MAE to
   the U-Net isolates the spatial denoising contribution from physics decoding quality.

---

## File Map

```
train_phys_recon.py             Synthetic trainer (v4.6): G0_NORM, scaled-sigmoid, masked_huber_tv
iscors_real_runner.ipynb        Phase-2 real-data notebook (v4.6 reference pipeline)
app.py                          Gradio inference frontend (HF Spaces)
datasets/phys_recon_dataset.py  G_empirical + G(0)=CV² norm, σ_G map, kurtosis artifact mask
models/pissl_tau_encoder.py     U-Net, scaled-sigmoid γ/α, τ-PE (v3.9), optional σ input (v4.1)
loss/phys_recon_loss.py         Fisher × Reliability weighted MSE; g0_norm + shape_only modes
utils/traditional_iscors.py     FFT autocorrelation, curve fitting, G_empirical map
utils/gpu_iscors_fit.py         GPU-batched classical (γ,α) fitter — ablation baseline + fast quasi-GT
utils/gpu_stics.py              GPU iMSD — independent α verifier (STICS spatial-spread projection)
utils/generate_test_video.py    Synthetic v1 (nested circles) + v2 (concentric rings)
```
