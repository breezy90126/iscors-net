# iSCORS-Net — Retrospective: what one iSCAT video actually yields, and where ML fits

This is the honest closure of the iSCORS-Net line. We set out to accelerate / improve
iSCAT anomalous-diffusion mapping with a self-supervised U-Net. The project's lasting value
is not the network — it is the **measured boundary of what a single iSCAT video yields**,
and a precise account of where machine learning can and cannot help. Both turned out to be
negative for this data — but they are *measured*, not assumed, and that is the contribution.

---

## 0. The premise was on the wrong axis: sparse-τ ≠ sparse frames

iSCORS-Net was framed as the FAST-style idea — "sparse sampling + a 2D network." But the
sparsity it actually used was **sparse τ lags**: compute the autocorrelation G(τ) at K≈10
chosen lags and feed those channels to the U-Net.

**That axis carries no cost and no noise, so there was nothing for the network to optimise.**
Each G(τ) channel is computed from **all T frames** → it is already clean (T=2000 → ~2 %
error even at the worst lag). "Sparse τ" is therefore just **feature selection on
fully-sampled, low-noise data**; it saves nothing in acquisition and creates no
ill-posedness. A learned prior only adds value when the estimator is **starved** (few
measurements / low SNR) — which sparse-τ never is.

The genuine FAST analog is **sparse frames**: use *fewer frames* (short T). *That* axis has
a real cost (acquisition time, photodamage, live-imaging speed) and a real noise problem
(few-frame G(τ) is noisy, worst at large τ), and a learned spatial prior could plausibly
help — validated against the full-T fit as ground truth. **iSCORS-Net never operated on
this axis.** It put ML where ML cannot win (clean, fully-sampled τ) instead of where it
might (starved frames).

> Bottom line of §0: the network was applied to the wrong sparsity axis from the start.
> The head-to-head (§3) is the empirical confirmation — it did not optimise anything.

---

## 1. Projection inventory & the acquisition-regime mismatch

A single (T,H,W) iSCAT video is a set of lossy projections. We ran each one and let it
report honestly whether it has signal:

| Projection | Physical question | Verdict on this data |
|---|---|---|
| **ACF / TICS** (per-pixel temporal) | how fast / what kind of motion (γ, α) | ✅ γ map + apparent global α ≈ 0.6 |
| **STICS / iMSD** (spatio-temporal) | spatial transport v, α from spread | ❌ **sub-PSF** — σ²(τ) never grows above the PSF floor |
| **χ4** (four-point) | dynamic heterogeneity (cooperative τ*) | ❌ **no peak**; real χ4 below the time-shuffle floor → homogeneous |
| **amplitude / density** (G(0)=CV², N&B) | how much / how many / aggregation | ⭐ robust, **unexploited** (normalised away) |

**The two empty projections are an acquisition-regime mismatch, not a method failure.**
The data is ≈5000 fps × 1 s, ~130 nm/px (PSF ≈ a few px). With chromatin D ~ 10⁻³–10⁻²
µm²/s the RMS displacement over the whole second is ≈ 60–200 nm ≈ **< 1 PSF** → STICS sees
no spreading. Cooperative / LLPS-type timescales are τ* ~ tens of s – minutes **≫ the 1 s
window** → χ4 has no peak to find. **SNR is not the limit** — averaging 10 frames to raise
SNR left both results unchanged; the limits are **window length** and **spatial resolution**,
which SNR does not touch.

So the fast decorrelation the ACF measures (⟨G⟩ decaying over ms, α ≈ 0.6) is **fast
in-place fluctuation, not chromatin translation**. The slow transport and heterogeneity
simply fall outside this short, high-fps, diffraction-limited window. No analysis or ML
recovers what was never sampled.

---

## 2. The deliverable: GPU classical fit (γ + apparent α≈0.6 + density)

What the ACF *does* yield, computed classically on the GPU in seconds:

- **γ map** — the diffusion-rate map; method-consistent across single-component, two-
  component, classical and (former) ML. The primary product.
- **apparent global α ≈ 0.6** — one cell-wide anomalous exponent, agreed by classical B
  (0.61), classical E (0.69) and the 1-component global-α net (0.63). Labelled *apparent /
  temporal-decorrelation* exponent: STICS could not verify whether it is genuine
  sub-diffusion or heterogeneity/confinement (the α/σ_D degeneracy is unbreakable from one
  ACF; a per-pixel α is **not** identifiable — high-R² regions can carry an arbitrary α).
- **density map (G(0) = CV²)** — the amplitude channel iSCORS normalises away. High-SNR,
  needs no fit, ≈ the clean iSCORS condensation map; "how much/how many", complementary to
  γ's "how fast". This is the cleanest single map the video affords.

`utils/gpu_iscors_fit.py` (`gpu_fit_maps`, `compute_density`) produces all of these; the
deliverable notebook `iscors_deliverable.ipynb` wraps them with a Gradio front-end.

---

## 3. The ML verdict: the U-Net is retired for γ (measured, not assumed)

ML/sparse-sampling adds value only where the classical estimator is **starved** (data/SNR)
or **absent** (no method). The per-pixel ACF at T=2000 is neither — well-fed and well-served
by classical fitting. The head-to-head confirms it.

**Head-to-head (matched spec: 1-component, global α), held-out checkerboard grids:**

| metric | floor A (1-param) | matched classical B | U-Net |
|---|---|---|---|
| self-consistency (gridA↔gridB) | 0.967 | **0.973** | **0.838** |
| vs independent quasi-GT | 0.947 | 0.959 | 0.834 |

**ML value = pM − p_matched = −0.135.** The network is **measurably *worse*** than a
per-pixel classical fit — less reproducible across independent noise — not merely
redundant. It is also slower for the per-video self-supervised use (minutes of training vs
seconds of classical fitting). The "hours" the README once attributed to traditional iSCORS
was a CPU/scipy per-pixel artifact; GPU-batched classical fitting is seconds.

Why ML had no room: γ at full T is not noise-limited (temporal averaging already denoises),
and α is not noise-limited but **information**-limited (the α/σ_D degeneracy) — ML denoising
cannot manufacture information that the projection does not contain. On the starved /
coherence-limited axes (§4), ML cannot reach either.

---

## 4. SOFI is blocked by coherence — a phase-loss inverse problem, not a math gap

The fast in-place fluctuation is SOFI's substrate, and SOFI's super-resolution (cumulant →
PSFⁿ) would, in principle, sharpen the effective PSF and reopen the sub-PSF spatial axis
that killed STICS. It does not transfer to dense cellular iSCAT:

- iSCAT is **coherent**: I = |E_ref + ΣE_k|². In the dense regime the cross-scatterer
  interference terms (speckle) are not negligible, and the intensity correlation mixes the
  field correlation g1 with anomalous terms (Siegert-type), so the cumulant is **not** a
  clean Σ of squared, positive PSFs.
- The field PSF is **complex / bipolar**, so cross-pixel cumulants **oscillate in sign** —
  the hallmark the prior full-formula attempt found.
- **No operator fixes it.** Removing the sign (square / magnitude, DDM-style) destroys the
  phase that super-resolution needs → you get q-space *dynamics* (DDM), not a sharpened
  image. Keeping the phase keeps the sign oscillation. You cannot have positive **and**
  super-resolving from coherent intensity.
- **weak-form / SINDy do not help**: they solve numerical-differentiation noise and
  dynamical-term selection. The SOFI wall is **lost phase** (a phase-retrieval inverse
  problem with a real null space), not a differentiation problem; integration-by-parts
  cannot recover discarded phase, and sparse regression cannot select information that is
  physically absent.

SOFI on iSCAT needs an **acquisition change** — measure the field (off-axis holography /
quantitative phase) or use **sparse** scatterers (single-particle iSCAT) — not a cleverer
algorithm on dense intensity data.

---

## 5. Unexploited channel & guidance for the next dataset

- **Unexploited now:** the **density / amplitude channel** (G(0)=CV², and Number & Brightness
  for count vs brightness / aggregation). Robust, classical, no ML — and it is what a clean
  condensation map actually is. Added to the deliverable.
- **To see chromatin slow dynamics** (transport, heterogeneity): acquire the *opposite*
  trade-off — **long duration (tens of s – minutes) at low fps** — accepting worse temporal
  resolution. The STICS-iMSD (`utils/gpu_stics.py`) and χ4 (`utils/gpu_chi4.py`) probes apply
  unchanged; the wall is the data regime, not the tooling.
- **To do fluctuation super-resolution:** **measure the field** (phase) or use **sparse
  labelling**. Then a field-based / sparse-phase-retrieval method (not naive SOFI) is viable.
- **Where ML would finally have a precondition:** short-T (sparse-frame) reconstruction
  validated against the full-T fit (efficiency, with GT), and noise-starved higher-order
  statistics (χ4) on long-duration data — i.e. the *frame* axis and the *no-good-classical*
  regimes, never the clean per-pixel ACF.

---

## Closure

For this video the single-cell extractable content is the **ACF (γ + apparent α≈0.6) plus
the density channel** — all produced by a seconds-fast GPU classical fit. The U-Net is
retired (measured worse, not just redundant). STICS, χ4 and SOFI are blocked by the
acquisition regime and by coherent phase loss, not by analysis quality. The deliverable is
small, fast, classical and honest; the scientific contribution is the *resolution-limit-aware
boundary* the multi-projection brief asked for — now drawn from measurement on real data.

See `EXPERIMENTS.md` for the full version history and per-decision evidence.
