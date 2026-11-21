"""Render the iSCORS (COBRI, transmission) iPSF with the ipsf_3dgs vector forward model
and save it for utils/ipsf_kernel.py  (-> notebook config IPSF_NPZ).

The ipsf_3dgs project models a REFLECTION spiral iSCAT (532 nm, NA 1.4, 72 nm, vortex
l=1, SLM). The iSCORS paper (Hsiao et al. 2024, Methods) is TRANSMISSION:
660 nm laser, OBJ2 oil NA 1.49, 42 nm/px, no vortex, and a 3 mm silver dot in a
plane conjugate to the OBJ2 back focal plane that passes ~1% of the light
(it attenuates the unscattered reference AND the low-k part of the scattered light).

What is switched, and why (all through existing OpticsSpec / params knobs, no edits
to ipsf_3dgs):
  wavelength / NA / pixel   660 nm / 1.49 / 42 nm
  n_medium                  1.38 (nucleoplasm; 1.33 for beads in water)
  l = 0                     no spiral phase plate
  aberr_z_const = -1        the particle-height phase. Reflection (default +1):
                            k z (n_med + k_z) = illumination up-trip + scattered
                            return. Transmission: scattered and transmitted
                            reference both travel forward, so only the defocus
                            difference survives: k z (k_z - n_med).
  slm_hole_*                re-used as the back-focal-plane dot: scattered-field
                            power x DOT_FLOOR inside |k| < DOT_RADIUS (1/um).
                            The reference attenuation only rescales contrast, which
                            the normalised kernel divides out.
  slm_mod_efficiency = 1    no SLM leakage
  image_rot = 0, pupil residual / Zernike off (the run89 aberration is the spiral
                            rig's, not the COBRI's)

DOT_RADIUS is an ESTIMATE: 3 mm dot, 1:1 relay (L3 = L4 = 200 mm), OBJ2 100x
(f = 1.8 mm) -> BFP radius f*NA = 2.68 mm -> dot covers sin(theta) < 0.56*NA, i.e.
|k| < 0.56*1.49/0.660 um = 1.26 1/um. Change it if the relay is not 1:1.

Usage (needs torch + the ipsf_3dgs package, e.g. on your PC or in Colab):
    python scripts/export_ipsf_3dgs_cobri.py --ipsf-root "C:/Users/silen/Desktop/claude code/ipsf by 3dgs" \
        --out ipsf_cobri.npz
then in the notebook config:  IPSF_NPZ = '/path/to/ipsf_cobri.npz'
"""
import argparse
import dataclasses
import math
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipsf-root", required=True, help="folder that contains ipsf_3dgs/")
    ap.add_argument("--out", default="ipsf_cobri.npz")
    ap.add_argument("--npix", type=int, default=256)
    ap.add_argument("--pixel-nm", type=float, default=42.0)
    ap.add_argument("--wavelength-nm", type=float, default=660.0)
    ap.add_argument("--na", type=float, default=1.49)
    ap.add_argument("--n-medium", type=float, default=1.38)
    ap.add_argument("--z-focal-nm", type=float, default=2000.0,
                    help="focal plane above the coverslip (inside the nucleus)")
    ap.add_argument("--dz-nm", type=float, nargs="+", default=[-300, -150, 0, 150, 300],
                    help="scatterer heights relative to the focal plane")
    ap.add_argument("--radius-nm", type=float, default=20.0,
                    help="scatterer radius (only scales the contrast; kernel is normalised)")
    ap.add_argument("--n-particle", type=float, default=1.45)
    ap.add_argument("--dot-radius-inv-um", type=float, default=1.26)
    ap.add_argument("--dot-floor", type=float, default=0.01, help="power transmission of the dot")
    ap.add_argument("--forward-mode", default="scalar", choices=["scalar", "vector"])
    args = ap.parse_args()

    sys.path.insert(0, args.ipsf_root)
    import torch
    from ipsf_3dgs.physics.forward import TrainableForward
    from ipsf_3dgs.physics.mie import alpha_rayleigh
    from ipsf_3dgs.synth.config import OpticsSpec
    from ipsf_3dgs.synth import optics as synth_optics

    o = dataclasses.replace(
        OpticsSpec(),
        npix=args.npix, pixel_size=args.pixel_nm * 1e-9, NA=args.na,
        wavelength=args.wavelength_nm * 1e-9, n_medium=args.n_medium,
        l=0, aberr_z_const=-1.0,
        slm_hole_inv_um=args.dot_radius_inv_um, slm_hole_mode="soft",
        slm_mod_efficiency=1.0, image_rot_deg=0.0, global_phase_rad=0.0,
        z_focal_nm=args.z_focal_nm, forward_mode=args.forward_mode,
        apply_polar_mod=False,
    )
    radius = args.radius_nm * 1e-9
    params = o.to_params(n_particle=complex(args.n_particle), diameter=2 * radius)
    params["slm_hole_floor"] = args.dot_floor          # dot passes ~1% power
    params["slm_hole_width_inv_um"] = 0.05             # sharp-edged metal film
    tf = TrainableForward(params)
    synth_optics.pin(tf, o)
    dev = getattr(tf, "device", torch.device("cpu"))

    alpha = complex(alpha_rayleigh(complex(args.n_particle), radius, args.n_medium))
    alpha_t = torch.tensor([[alpha]], dtype=torch.complex128, device=dev)
    stack = []
    with torch.no_grad():
        for dz in args.dz_nm:
            z = (args.z_focal_nm + dz) * 1e-9
            coords = torch.tensor([[[0.0, 0.0, z]]], dtype=torch.float64, device=dev)
            img = tf.render_contrast(coords, alpha_t)[0].detach().cpu().numpy()
            stack.append(img.astype(np.float64))
            r = np.hypot(*np.meshgrid(np.arange(img.shape[0]) - img.shape[0] / 2,
                                      np.arange(img.shape[1]) - img.shape[1] / 2))
            print(f"dz={dz:+6.0f} nm  contrast [{img.min():+.2e}, {img.max():+.2e}]  "
                  f"centre {img[r < 2].mean():+.2e}")
    stack = np.stack(stack)
    np.savez(args.out, ipsf=stack, z=(args.z_focal_nm + np.asarray(args.dz_nm)) * 1e-9,
             pixel_size=args.pixel_nm * 1e-9,
             meta=np.array(str({k: v for k, v in vars(args).items()})))
    print(f"wrote {args.out}: {stack.shape}")


if __name__ == "__main__":
    main()
