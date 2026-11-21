"""Export a single-particle iPSF stack from the full vector generator, for utils/ipsf_kernel.py.

Needs torch (run it in Colab / on the GPU box). Writes an .npz with
    ipsf       (nz, H, W) float64  iPSF contrast of ONE particle at the image centre
    z          (nz,)               particle heights (m)
    pixel_size ()                  camera pixel (m)

Then, numpy only:
    from utils.ipsf_kernel import load_ipsf_npz, ipsf_spectrum, ipsf_kernel
    ipsf, z, px = load_ipsf_npz("ipsf_stack.npz")
    q2, S = ipsf_spectrum(ipsf, px, bin_factor=BIN_FACTOR, z_weights=None, n_q2_bins=512)
    K, D = ipsf_kernel(RECON_TAUS, dt=FRAME_TIME, q2=q2, S=S, T=N_FRAMES)

Usage:
    python scripts/export_ipsf_stack.py --params params.json --z 30e-9 70e-9 120e-9 \
        --out ipsf_stack.npz

`params.json` is the legacy simulation config (its `optical_params` block); it is not on
this branch — take it from legacy/deep-learning-stics or pass your own.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.xyt_dataset_generator import EnhancedDifferentiableForwardModel  # noqa: E402


def build_model(optical_params, device):
    p = dict(optical_params)
    p.setdefault("polarization",
                 torch.tensor([1.0, 1j], dtype=torch.complex128) / math.sqrt(2.0))
    p.setdefault("qwp_angle", math.pi)
    return EnhancedDifferentiableForwardModel(p).to(device)


@torch.no_grad()
def single_ipsf(model, z, device):
    """Same chain as simulate_with_background, without background or noise."""
    coords = torch.tensor([[[0.0, 0.0, float(z)]]], dtype=torch.float64, device=device)
    E_s = model.create_scattered_field(coords)
    E_s = model.apply_polarization_modulation(E_s)
    E_s = model.apply_phase_mask(E_s)
    ipsf = model.compute_ipsf(E_s, particle_coords=coords)
    return ipsf[0].detach().cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--z", type=float, nargs="+", default=[70e-9])
    ap.add_argument("--out", default="ipsf_stack.npz")
    args = ap.parse_args()

    with open(args.params) as f:
        cfg = json.load(f)
    optical = cfg.get("optical_params", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(optical, device)

    stack = np.stack([single_ipsf(model, z, device) for z in args.z])
    np.savez(args.out, ipsf=stack, z=np.asarray(args.z),
             pixel_size=float(optical.get("cam_pixelsize", 72e-9)))
    print(f"wrote {args.out}: ipsf {stack.shape}, z = {args.z}")


if __name__ == "__main__":
    main()
