"""Generate ipsf_cobri_xsec.ipynb (stdlib only).

Standalone Colab notebook: render the iSCORS (COBRI, transmission) iPSF with the
ipsf_3dgs forward model, check it against scattering theory for samples of different
polarizability, then export the iPSF stack for the D6 kernel.
"""
import json
import os

CELLS = []


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src.replace("~~~", chr(34) * 3)))


md(r"""# 穿透式（COBRI）iPSF 驗證：不同極化率樣品的散射截面積

用 `ipsf_3dgs` 的 forward model（你的 3DGS 專案），以參數切成 iSCORS 論文的**穿透式 COBRI**：
660 nm、OBJ2 NA 1.49、42 nm/px、無螺旋相位板、背焦面中心銀點衰減片（~1% 穿透）。
然後對**一組不同極化率的樣品**檢查：

| 檢查 | 預期 | 為什麼 |
|---|---|---|
| A. 暗場功率 ∫\|E_s\|² vs 散射截面積 σ_sca | log-log 斜率 = 1，比值跨樣品為常數 | forward 的 \|E_s0\|² ∝ k⁴\|α\|²/6π = σ_sca |
| B. 干涉對比振幅 vs \|α\| | 斜率 = 1 | iSCAT 的交叉項 ∝ \|E_s\| ∝ \|α\| |
| C. 自己寫的 Mie σ vs Rayleigh σ | 小粒子時一致（<1%）；100 nm PS 開始偏離 | 確認 Mie 實作與 Rayleigh 極限 |
| D. 論文的數字 | 50 nm、n=1.49 的染色質團 → 對比 ≈ 0.01（論文 Discussion 的偵測極限） | 唯一在正文裡的絕對量 |

正文只有 D 這一個絕對數字。SI（Supplementary Fig. 4、8）的奈米粒子校正數據請填進 config 的
`PAPER_SI`，notebook 會一起比對。

最後一格把染色質團的 iPSF 堆疊存成 `.npz`，填進 `iscors_one_shot_diagnostics.ipynb` 的 `IPSF_NPZ` 即可。
所有圖（png）、log、`results.json` 都存在 Drive 的 `OUT_DIR`。

**幾何切換只用參數，不改 ipsf_3dgs 的程式**（細節見 `scripts/export_ipsf_3dgs_cobri.py`）：
`l=0`、`aberr_z_const=-1`（穿透：相位 z(k_z−n)，反射是 z(k_z+n)）、`slm_hole` 當衰減點、像差歸零。
""")

code(r"""# ── setup: Drive, both repos, deps ──
import os, sys, subprocess
try:
    from google.colab import drive; drive.mount('/content/drive', force_remount=False)
except Exception: pass

def _clone(url, branch, path):
    if os.path.isdir(path):
        for c in (['git', '-C', path, 'fetch', 'origin'], ['git', '-C', path, 'checkout', branch],
                  ['git', '-C', path, 'pull', 'origin', branch]): subprocess.run(c, check=False)
    else:
        subprocess.run(['git', 'clone', '--branch', branch, url, path], check=False)
    return os.path.isdir(path)

ISCORS_DIR = '/content/iscors-net'
_clone('https://github.com/breezy90126/iscors-net.git', 'claude/project-thread-kkiy52', ISCORS_DIR)

# ipsf_3dgs package: from the uploaded ipsf_3dgs.zip (Drive, or the Colab Files panel);
# falls back to git clone of breezy90126/ipsf-3dgs (private repo -> Colab Secret GH_TOKEN)
IPSF_ZIP_CANDIDATES = ['/content/ipsf_3dgs.zip', '/content/drive/MyDrive/iscors_test/ipsf_3dgs.zip']
IPSF_DIR = '/content/ipsf_pkg'
import zipfile
zp = next((p for p in IPSF_ZIP_CANDIDATES if os.path.exists(p)), None)
if zp:
    os.makedirs(IPSF_DIR, exist_ok=True)
    with zipfile.ZipFile(zp) as z: z.extractall(IPSF_DIR)
    print('ipsf_3dgs from', zp)
else:
    tok = None
    try:
        from google.colab import userdata; tok = userdata.get('GH_TOKEN')
    except Exception: pass
    url = (f'https://{tok}@github.com/breezy90126/ipsf-3dgs.git' if tok
           else 'https://github.com/breezy90126/ipsf-3dgs.git')
    IPSF_DIR = '/content/ipsf-3dgs'
    if not _clone(url, 'phase6-real-data-inversion', IPSF_DIR):
        raise SystemExit('upload ipsf_3dgs.zip to /content or MyDrive/iscors_test/ first')
subprocess.run(['pip', 'install', '-q', 'tifffile', 'tqdm', 'pillow'], check=False)
sys.path.insert(0, IPSF_DIR); sys.path.insert(0, ISCORS_DIR)

import json, math, time, builtins, dataclasses
import numpy as np, matplotlib.pyplot as plt, torch
from scipy.special import spherical_jn, spherical_yn
from ipsf_3dgs.physics.forward import TrainableForward
from ipsf_3dgs.physics.mie import alpha_rayleigh
from ipsf_3dgs.synth.config import OpticsSpec
from ipsf_3dgs.synth import optics as synth_optics
import utils.ipsf_kernel as ik
print('setup ok, torch', torch.__version__, '| cuda' if torch.cuda.is_available() else '| cpu')""")

code(r"""# ── config (EDIT) ──
OUT_DIR = '/content/drive/MyDrive/iscors_test/ipsf_xsec/' + time.strftime('%Y%m%d_%H%M')

# COBRI optics (paper Methods)
WAVELENGTH = 660e-9; NA = 1.49; PIXEL = 42e-9; NPIX = 256
Z_FOCAL_NM = 2000.0                    # focal plane above the coverslip
DZ_NM = (-300, -150, 0, 150, 300)      # heights scanned around focus (contrast peak = max over these)
DOT_RADIUS_INV_UM = 1.26               # estimate: 3 mm dot, 1:1 relay, 100x f=1.8 mm BFP
DOT_FLOOR = 0.01                       # dot power transmission (~1%, paper Methods)
FORWARD_MODE = 'scalar'                # or 'vector'

# samples of different polarizability: (name, diameter_nm, n_particle, n_medium)
SAMPLES = [
    ('PS 20 nm',            20, 1.59, 1.33),
    ('PS 40 nm',            40, 1.59, 1.33),
    ('PS 60 nm',            60, 1.59, 1.33),
    ('PS 80 nm',            80, 1.59, 1.33),
    ('PS 100 nm',          100, 1.59, 1.33),
    ('chromatin 50 nm',     50, 1.49, 1.38),   # the paper's detection-limit object
    ('chromatin 100 nm',   100, 1.49, 1.38),
]
CHROMATIN_FOR_KERNEL = 'chromatin 50 nm'      # whose iPSF is exported for D6

# numbers from the paper. Main text (Discussion): a ~50 nm nanocluster with n = 1.49 is
# at the noise floor C ~ 0.01. Add SI values here as  name: dict(sigma_nm2=..., contrast=...)
PAPER = {'chromatin 50 nm': dict(contrast=0.01)}
PAPER_SI = {}                                  # e.g. 'PS 100 nm': dict(contrast=0.05)

# ── save every figure + all printed text ──
os.makedirs(OUT_DIR, exist_ok=True)
_show = plt.show; _N = [0]
def _save_show(*a, **kw):
    if plt.gcf().get_axes():
        _N[0] += 1; fn = os.path.join(OUT_DIR, f'fig{_N[0]:02d}.png')
        plt.gcf().savefig(fn, dpi=150, bbox_inches='tight'); builtins.print('  [saved]', fn)
    return _show(*a, **kw)
plt.show = _save_show
_print = builtins.print
def print(*a, **kw):
    _print(*a, **kw)
    with open(os.path.join(OUT_DIR, 'log.txt'), 'a') as f:
        _print(*a, **{k: v for k, v in kw.items() if k != 'file'}, file=f)
RESULT = {}
print('outputs ->', OUT_DIR)""")

code(r"""# ── helpers: COBRI forward, +/-alpha split, Mie / Rayleigh cross sections ──
def build_cobri(diameter_m, n_p, n_med):
    o = dataclasses.replace(
        OpticsSpec(), npix=NPIX, pixel_size=PIXEL, NA=NA, wavelength=WAVELENGTH, n_medium=n_med,
        l=0, aberr_z_const=-1.0, slm_hole_inv_um=DOT_RADIUS_INV_UM, slm_hole_mode='soft',
        slm_mod_efficiency=1.0, image_rot_deg=0.0, global_phase_rad=0.0,
        z_focal_nm=Z_FOCAL_NM, forward_mode=FORWARD_MODE, apply_polar_mod=False)
    params = o.to_params(n_particle=complex(n_p), diameter=diameter_m)
    params['slm_hole_floor'] = DOT_FLOOR
    params['slm_hole_width_inv_um'] = 0.05
    tf = TrainableForward(params); synth_optics.pin(tf, o)
    return tf

@torch.no_grad()
def split_render(tf, alpha, z_m):
    ~~~contrast C(+a), C(-a) -> interferometric X = (C+ - C-)/2, dark-field D = (C+ + C-)/2.
    (I - I_ref)/I_ref = 2Re(E_r* E_s)/|E_r|^2 + |E_s|^2/|E_r|^2 ; flipping alpha flips only the first.~~~
    dev = getattr(tf, 'device', torch.device('cpu'))
    c = torch.tensor([[[0.0, 0.0, z_m]]], dtype=torch.float64, device=dev)
    out = []
    for s in (1, -1):
        a = torch.tensor([[s * alpha]], dtype=torch.complex128, device=dev)
        out.append(tf.render_contrast(c, a)[0].detach().cpu().numpy().astype(np.float64))
    return (out[0] - out[1]) / 2, (out[0] + out[1]) / 2

def sigma_rayleigh(d, n_p, n_med):
    k = 2 * np.pi * n_med / WAVELENGTH; r = d / 2; m2 = (n_p / n_med) ** 2
    return 8 * np.pi / 3 * k ** 4 * r ** 6 * abs((m2 - 1) / (m2 + 2)) ** 2

def sigma_mie(d, n_p, n_med):
    k = 2 * np.pi * n_med / WAVELENGTH; x = k * d / 2; m = n_p / n_med; mx = m * x
    nmax = int(np.ceil(x + 4 * x ** (1 / 3) + 2)); n = np.arange(1, nmax + 1)
    jx, jdx = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    yx, ydx = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
    jm, jdm = spherical_jn(n, mx), spherical_jn(n, mx, derivative=True)
    psi, dpsi = x * jx, jx + x * jdx
    xi, dxi = x * (jx + 1j * yx), (jx + 1j * yx) + x * (jdx + 1j * ydx)
    psim, dpsim = mx * jm, jm + mx * jdm
    an = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
    bn = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
    return 2 * np.pi / k ** 2 * np.sum((2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2))

# self-check: Mie -> Rayleigh for a tiny sphere
r_chk = sigma_mie(5e-9, 1.59, 1.33) / sigma_rayleigh(5e-9, 1.59, 1.33)
print(f'Mie/Rayleigh at d=5 nm: {r_chk:.4f}  (should be 1.000)'); assert abs(r_chk - 1) < 0.01""")

code(r"""# ── run every sample ──
rows = []
for name, d_nm, n_p, n_med in SAMPLES:
    d = d_nm * 1e-9
    tf = build_cobri(d, n_p, n_med)
    alpha = complex(alpha_rayleigh(complex(n_p), d / 2, n_med))
    Xs, Ds = [], []
    for dz in DZ_NM:
        X, D = split_render(tf, alpha, (Z_FOCAL_NM + dz) * 1e-9); Xs.append(X); Ds.append(D)
    Xs, Ds = np.stack(Xs), np.stack(Ds)
    i0 = list(DZ_NM).index(0) if 0 in DZ_NM else len(DZ_NM) // 2
    x_peak = float(np.abs(Xs).max())                        # max over z and pixels
    df_pow = float(Ds[i0].sum() * PIXEL ** 2)               # collected dark-field, in-focus
    row = dict(name=name, d_nm=d_nm, n_p=n_p, n_med=n_med, alpha=abs(alpha),
               sig_ray=sigma_rayleigh(d, n_p, n_med), sig_mie=sigma_mie(d, n_p, n_med),
               x_peak=x_peak, x_peak_cobri=x_peak / math.sqrt(DOT_FLOOR),
               df_pow=df_pow, X=Xs, D=Ds)
    rows.append(row)
    print(f'{name:18s} |alpha|={abs(alpha):.3e} m^3  sigma_Ray={row["sig_ray"]*1e18:9.3e} nm^2  '
          f'sigma_Mie={row["sig_mie"]*1e18:9.3e} nm^2  X_peak={x_peak:.3e}  '
          f'(x1/sqrt(dot) = {row["x_peak_cobri"]:.3e})  darkfield={df_pow:.3e}')
del tf""")

code(r"""# ── checks A-C ──
a   = np.array([r['alpha'] for r in rows]); sr = np.array([r['sig_ray'] for r in rows])
sm  = np.array([r['sig_mie'] for r in rows]); xp = np.array([r['x_peak'] for r in rows])
dfp = np.array([r['df_pow'] for r in rows]); names = [r['name'] for r in rows]

slope_A = np.polyfit(np.log10(sm), np.log10(np.abs(dfp)), 1)[0]
ratio_A = dfp / sm
slope_B = np.polyfit(np.log10(a), np.log10(xp), 1)[0]
mie_ray = sm / sr
print(f'A. dark-field power vs sigma_Mie : slope {slope_A:.3f} (expect 1.000); '
      f'power/sigma spread {ratio_A.std()/ratio_A.mean()*100:.1f}% of mean')
print(f'B. interferometric peak vs |alpha|: slope {slope_B:.3f} (expect 1.000)')
print('C. sigma_Mie / sigma_Rayleigh   :', ', '.join(f'{n} {v:.3f}' for n, v in zip(names, mie_ray)))
RESULT.update(slope_A=slope_A, slope_B=slope_B, power_over_sigma_cv=float(ratio_A.std() / ratio_A.mean()),
              mie_over_rayleigh=dict(zip(names, mie_ray.tolist())))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].loglog(sm * 1e18, np.abs(dfp), 'o'); ax[0].set_xlabel('sigma_Mie [nm^2]'); ax[0].set_ylabel('model dark-field power')
ax[0].set_title(f'A: slope {slope_A:.3f} (expect 1)')
ax[1].loglog(a, xp, 'o'); ax[1].set_xlabel('|alpha| [m^3]'); ax[1].set_ylabel('interferometric peak')
ax[1].set_title(f'B: slope {slope_B:.3f} (expect 1)')
ax[2].semilogx(np.array([r['d_nm'] for r in rows]), mie_ray, 'o'); ax[2].axhline(1, color='k', lw=0.6)
ax[2].set_xlabel('diameter [nm]'); ax[2].set_ylabel('sigma_Mie / sigma_Rayleigh'); ax[2].set_title('C: Rayleigh limit')
for i, n in enumerate(names):
    ax[0].annotate(n, (sm[i] * 1e18, abs(dfp[i])), fontsize=7); ax[1].annotate(n, (a[i], xp[i]), fontsize=7)
plt.tight_layout(); plt.show()""")

code(r"""# ── check D: against the paper ──
ok_all = True
for name, ref in {**PAPER, **PAPER_SI}.items():
    r = next((r for r in rows if r['name'] == name), None)
    if r is None:
        print(f'{name}: not in SAMPLES, skipped'); continue
    if 'contrast' in ref:
        model = r['x_peak_cobri']; f = model / ref['contrast']
        ok = 1 / 3 <= f <= 3
        print(f'D. {name}: model contrast {model:.4f} (reference attenuated by the dot) vs paper '
              f'{ref["contrast"]:.4f}  -> ratio {f:.2f}  {"OK (within 3x)" if ok else "MISMATCH"}')
        ok_all &= ok
    if 'sigma_nm2' in ref:
        f = r['sig_mie'] * 1e18 / ref['sigma_nm2']; ok = 0.8 <= f <= 1.25
        print(f'D. {name}: sigma_Mie {r["sig_mie"]*1e18:.3e} nm^2 vs paper {ref["sigma_nm2"]:.3e} -> {f:.2f}'
              f'  {"OK" if ok else "MISMATCH"}')
        ok_all &= ok
print('\nNOTE: the paper contrast is an order-of-magnitude detection-limit statement (C ~ 0.01), and the'
      '\nmodel contrast depends on the dot transmission and the focus offset - a factor ~3 either way is'
      '\nthe honest tolerance. Fill PAPER_SI from Supplementary Figs 4/8 for a sharper test.')
RESULT['paper_checks_ok'] = bool(ok_all)

fig, ax = plt.subplots(2, len(rows), figsize=(2.4 * len(rows), 5))
for j, r in enumerate(rows):
    i0 = list(DZ_NM).index(0) if 0 in DZ_NM else len(DZ_NM) // 2
    c = NPIX // 2; w = 24
    v = np.abs(r['X'][i0]).max()
    ax[0, j].imshow(r['X'][i0][c-w:c+w, c-w:c+w], cmap='RdBu_r', vmin=-v, vmax=v); ax[0, j].set_title(r['name'], fontsize=8)
    ax[1, j].plot(np.array(DZ_NM), [r['X'][i][c, c] for i in range(len(DZ_NM))], 'o-'); ax[1, j].set_xlabel('dz [nm]', fontsize=8)
    ax[0, j].axis('off')
ax[1, 0].set_ylabel('centre contrast (raw)')
plt.suptitle('in-focus interferometric iPSF (top) and centre contrast vs height (bottom)', fontsize=10)
plt.tight_layout(); plt.show()""")

code(r"""# ── export the chromatin iPSF stack for the D6 kernel ──
r = next(r for r in rows if r['name'] == CHROMATIN_FOR_KERNEL)
stack = r['X'] + r['D']                                  # full contrast (cross + dark-field)
npz = os.path.join(OUT_DIR, 'ipsf_cobri.npz')
np.savez(npz, ipsf=stack, z=(Z_FOCAL_NM + np.asarray(DZ_NM)) * 1e-9, pixel_size=PIXEL)
q2, S = ik.ipsf_spectrum(stack, PIXEL, bin_factor=2, n_q2_bins=512)
q_eff = math.sqrt(ik.effective_q2(q2, S))
q2s, Ss = ik.ipsf_spectrum(ik.scalar_ipsf_stack([(Z_FOCAL_NM + dz) * 1e-9 - Z_FOCAL_NM * 1e-9 for dz in DZ_NM],
                                                 n_pix=NPIX, pixel_size=PIXEL, wavelength=WAVELENGTH, NA=NA,
                                                 n_medium=1.38, geometry='transmission'),
                           PIXEL, bin_factor=2, n_q2_bins=512)
q_eff_s = math.sqrt(ik.effective_q2(q2s, Ss))
print(f'wrote {npz}')
print(f'effective q: 3DGS COBRI {q_eff*1e-6:.2f} rad/um  vs scalar model {q_eff_s*1e-6:.2f} rad/um'
      f'  -> D from the old D6 run scales by {(q_eff_s/q_eff)**2:.2f}x')
print(f"\nIn iscors_one_shot_diagnostics.ipynb set:  IPSF_NPZ = '{npz}'")
RESULT.update(ipsf_npz=npz, q_eff_3dgs=q_eff, q_eff_scalar=q_eff_s, D_rescale_vs_scalar=(q_eff_s / q_eff) ** 2)
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(RESULT, f, indent=1, default=float)
_print('[saved]', os.path.join(OUT_DIR, 'results.json'))""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {"colab": {"provenance": []},
                 "accelerator": "GPU",
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ipsf_cobri_xsec.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "cells:", len(CELLS))
