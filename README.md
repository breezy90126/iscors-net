# iSCORS-Net

Classical (no-ML) GPU-batched iSCORS analysis of iSCAT video: per-pixel autocorrelation
G(τ) fitting to recover γ (diffusion) and derived condensation/reliability maps, plus a
Colab deliverable notebook that runs the full pipeline end to end.

**Live demo:** [huggingface.co/spaces/breezy90126/iscors-viewer](https://huggingface.co/spaces/breezy90126/iscors-viewer)
— upload a video, pick one of five axis-pair groups, get the left/right map pair.

```
G(τ) = A / (1 + γ · τ^α)
```

## What's here

- `iscors_deliverable.ipynb` — the runnable deliverable. Loads a video, computes the
  per-pixel maps, and renders the paper-replication / reliability-axis / ICA sections
  plus a dropdown viewer.
- `utils/gpu_iscors_fit.py` — GPU-batched classical (γ, α) fitter and the streaming ACF
  loader the notebook depends on. No neural network involved — this is direct
  optimisation of the physics model per pixel.
- `utils/traditional_iscors.py` — reference CPU/`scipy.optimize.curve_fit` implementation
  of the same fit, used for cross-checks.
- `data/condensation_mask.tif` — nucleus/condensation mask used by the deliverable.
- `papers/PMC11196589.pdf` — Hsiao *et al.*, "Probing chromatin condensation dynamics in
  live cells using interferometric scattering correlation spectroscopy," *Communications
  Biology* 7:763 (2024), doi:10.1038/s42003-024-06457-2. Section 1 of the notebook
  replicates its V_DLS–1/D* slope-3 condensation projection; Sections 2-3 are extensions
  beyond the paper (see `RETROSPECTIVE_session.md`).
- `app.py` — Gradio front-end for the same pipeline (upload a video, pick one of the
  five axis-pair groups), deployed as a Hugging Face Space:
  [breezy90126/iscors-viewer](https://huggingface.co/spaces/breezy90126/iscors-viewer).

## History note

This branch was split off from the project's earlier deep-learning line (a
self-supervised U-Net trained to predict γ/α from sparse τ sampling, plus STICS/χ4
diagnostics and synthetic-video generation). That work — and everything that depended on
it (`models/`, `datasets/`, `loss/`, `train*.py`, the runner notebooks, etc.) — lives on
the `legacy/deep-learning-stics` branch. It was set aside because the classical
per-pixel fit already gives the maps this project needs; the network didn't add value
over it. This branch keeps only what `iscors_deliverable.ipynb` actually runs.

## Setup

```bash
pip install -r requirements.txt
```

Then open `iscors_deliverable.ipynb`, set the paths in the config cell (video zip/tif/npz,
mask), and run top to bottom.
