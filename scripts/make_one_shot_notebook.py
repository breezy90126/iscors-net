"""Generate iscors_one_shot_diagnostics.ipynb (stdlib only)."""
import json

CELLS = []


def md(src):
    CELLS.append(("markdown", src.replace("~~~", chr(34)*3)))


def code(src):
    CELLS.append(("code", src.replace("~~~", chr(34)*3)))


# ─────────────────────────────── cell 0 ────────────────────────────────────
md(r"""# iSCORS 一次性診斷 —— 你在取樣動力學，還是在取樣混疊？

這份 notebook 把**一次 Colab run** 花在「什麼東西可能讓現有的所有軸失效」上，
而不是花在「再多算一條軸」。五個診斷共用**同一次載入**，依重要性排序。

出發點是 iSCAT 的物理：訊號是參考光與散射光的干涉，交叉項 ∝ cos(φ)，φ = 4πnz/λ。
代入 λ≈520 nm、n≈1.38：

| 軸向位移 | 後果 |
|---|---|
| ~47 nm | 對比由最大掉到零 |
| ~190 nm | 走完一整個干涉條紋 |

染色質的軸向漲落本來就在數十到數百 nm，**所以這是一個相位量測，不是強度量測**。

## 五個診斷

| | 測什麼 | 為什麼排這個順序 |
|---|---|---|
| **D1** | 時間功率譜（global vs local） | 唯一能判定**混疊**的診斷；混疊的話後處理救不回來 |
| **D2** | 每個 pixel 的 kurtosis | 相位有沒有繞過條紋；繞過的話該區 (γ,α) 無意義 |
| **D3** | 模型形式：代數 vs stretched exponential | 若 stretched 勝出，α 可直接讀、D_z 有物理單位 |
| **D4** | finite-T 偏差直接量測 | 把先前的模擬推論降級成量測 |
| **D5** | bulk 速率譜 | 前四項的前提沒問題才有意義 |

## 事前登記的預測（跑之前先寫下來）

| 預測 | 信心 |
|---|---|
| PSD 有至少一條銳線，且以 global mode 為主 | ~80% |
| 高頻白噪聲佔 variance 的 30–70% | ~70% |
| **沒有**混疊（高頻看得到白平台） | ~60% |
| kurtosis 大多 ≈3，但有一撮明顯 <3 | ~55% |
| stretched exponential 至少跟代數形式一樣好 | ~55% |
| bulk 速率譜單峰 | ~75% |
| finite-T 偏差比模擬預估的小 | ~65% |

最後一格會自動把量到的結果跟這張表對照。**預測錯了才有資訊量**，不要事後改預測。

## 怎麼讀

- 每個診斷自己印 `VERDICT:` 行，不需要看圖也能判讀。
- 每格都包在 `try/except` 裡：**一格掛掉不會毀掉整次 run**（這是一次性 run 的重點）。
- 背景推導見 `docs/laplace_projection_alpha.md`。
""")

# ─────────────────────────────── cell 1 ────────────────────────────────────
code(r"""# ── setup: clone/update repo from GitHub, mount Drive, install deps ──
import os, sys, subprocess
REPO = 'https://github.com/breezy90126/iscors-net.git'
BRANCH = 'claude/laplace-projection-alpha-ocpbfu'
REPO_DIR = '/content/iscors-net'
try:
    from google.colab import drive; drive.mount('/content/drive', force_remount=False)
except Exception: pass
if os.path.isdir(REPO_DIR):
    for c in (['git', '-C', REPO_DIR, 'fetch', 'origin'], ['git', '-C', REPO_DIR, 'checkout', BRANCH],
              ['git', '-C', REPO_DIR, 'pull', 'origin', BRANCH]): subprocess.run(c, check=False)
else:
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO, REPO_DIR], check=False)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
subprocess.run(['pip', 'install', '-q', 'tifffile', 'scipy', 'scikit-learn'], check=False)
print('setup done:', os.getcwd())

# ── imports used throughout ──
import gc, time, zipfile, traceback, numpy as np, matplotlib.pyplot as plt, tifffile
from scipy.ndimage import zoom, gaussian_filter, median_filter
from utils.gpu_iscors_fit import (compute_density, compute_g_norm_torch, gpu_fit_maps,
                                  vdls_amplitude)
import utils.laplace_spectrum as lsp

RESULT = {}          # every diagnostic writes its verdict here; last cell tabulates
def guard(name):
    ~~~Decorator: a failing diagnostic must not abort the one-shot run.~~~
    def deco(fn):
        try:
            fn()
        except Exception:
            RESULT[name] = {'verdict': 'ERROR', 'detail': traceback.format_exc(limit=3)}
            print(f'!! {name} failed (run continues):'); traceback.print_exc(limit=3)
    return deco
print('imports ok')""")

# ─────────────────────────────── cell 2 ────────────────────────────────────
code(r"""# ── config (EDIT paths + the acquisition block) ──
ZIP_PATH    = '/content/drive/MyDrive/iscors_test/large_file.zip'   # .zip or a .tif
VIDEO_FNAME = 'COBRI_rarw_video.tif'
EXTRACT_DIR = '/content/real_data'
MASK_PATH   = 'data/condensation_mask.tif'
N_FRAMES = 5000; BIN_FACTOR = 2; RECON_TAUS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
GAMMA_SCALE = 2.0

# ── acquisition / optics (EDIT — D1 and D3 read these) ──
FRAME_RATE_HZ = None      # None -> frequencies reported in cycles/frame (all verdicts still valid)
WAVELENGTH_NM = 520.0     # illumination wavelength
N_MEDIUM      = 1.38      # refractive index of the nucleoplasm

# ── diagnostic knobs ──
N_PSD_PX   = 2000         # pixels sampled for the local power spectrum
LINE_RATIO = 5.0          # a PSD peak this far above the local baseline counts as a "line"
KURT_LOW   = 2.5          # below this, the interference has likely wrapped through a fringe

Q_Z = 4 * np.pi * N_MEDIUM / (WAVELENGTH_NM * 1e-3)     # rad/um, axial interference wavevector
print(f'config ready.  q_z = {Q_Z:.1f} rad/um  ->  one fringe every '
      f'{2 * np.pi / Q_Z * 1000:.0f} nm of axial motion')
print('frame rate:', FRAME_RATE_HZ or '(unset -> cycles/frame)')""")

# ─────────────────────────────── cell 3 ────────────────────────────────────
code(r"""# ── loaders (same full-load convention as iscors_deliverable.ipynb) + diagnostic helpers ──
def load_video(path, fname=None, n_frames=2000, bin_factor=2, start_frame=0):
    if str(path).lower().endswith('.zip'):
        ed = globals().get('EXTRACT_DIR', '/content/real_data'); os.makedirs(ed, exist_ok=True)
        def _find(root, name):
            for dp, _, fs in os.walk(root):
                if name in fs: return os.path.join(dp, name)
            return None
        vp = _find(ed, fname)
        if not vp:
            with zipfile.ZipFile(path) as z: z.extractall(ed)
            vp = _find(ed, fname)
        assert vp, f'{fname} not found in zip'
    else:
        vp = path
    H0, W0 = tifffile.imread(vp, key=0).shape
    Hb, Wb = H0 // bin_factor, W0 // bin_factor
    with tifffile.TiffFile(vp) as tf:
        try:    total = int(tf.series[0].shape[0])
        except Exception: total = len(tf.pages)
    start_frame = max(0, min(int(start_frame), total))
    n = max(0, min(int(n_frames), total - start_frame)); Hc, Wc = Hb * bin_factor, Wb * bin_factor
    raw = np.empty((n, Hb, Wb), np.float32)
    for s in range(0, n, 100):
        e = min(s + 100, n)
        ch = tifffile.imread(vp, key=range(start_frame + s, start_frame + e)).astype(np.float32)
        raw[s:e] = ch[:, :Hc, :Wc].reshape(e - s, Hb, bin_factor, Wb, bin_factor).mean((2, 4))
    print(f'[load] {os.path.basename(vp)}: frames [{start_frame},{start_frame+n})/{total}  binned {raw.shape}')
    return raw

def preprocess(raw):
    ff = raw / (np.median(raw, axis=0)[None] + 1e-10)
    for t in range(len(ff)): ff[t] /= (gaussian_filter(ff[t], 4) + 1e-10)
    return ff

def load_mask(shape):
    mk = np.asarray(tifffile.imread(MASK_PATH)).squeeze()
    if mk.ndim == 3: mk = mk[..., 0]
    if mk.shape != shape: mk = zoom(mk, (shape[0] / mk.shape[0], shape[1] / mk.shape[1]), order=0)
    bd = np.zeros(shape, bool); bd[0, :] = bd[-1, :] = bd[:, 0] = bd[:, -1] = True
    return mk == min(np.unique(mk), key=lambda z: (mk[bd] == z).mean())   # interior = nucleus

# ── D1 helper: one-sided PSD, split into common-mode and per-pixel parts ──
def psd_bundle(arr, nuc, npx=2000, seed=0, chunk=200):
    ~~~Global (spatial-mean) and local (pixel-minus-global) temporal power spectra.

    Normalised so that sum(P) * df = variance, with df = 1/T (cycles/frame).
    Splitting matters: instrumental lines (vibration, laser, mains) are COMMON to
    every pixel and land in P_glob; sample dynamics are independent per pixel and
    land in P_loc. The ratio is what separates the two.
    ~~~
    T = arr.shape[0]
    ys, xs = np.nonzero(nuc)
    rs = np.random.RandomState(seed)
    sel = rs.choice(len(ys), size=min(npx, len(ys)), replace=False)
    ys_s, xs_s = ys[sel], xs[sel]
    gts = np.empty(T, np.float64)
    loc = np.empty((T, len(sel)), np.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        blk = arr[s:e]
        gts[s:e] = blk[:, ys, xs].mean(axis=1)
        loc[s:e] = blk[:, ys_s, xs_s]
    def _psd(x):
        x = np.asarray(x, np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        F = np.fft.rfft(x, axis=0)
        return (np.abs(F) ** 2) * (2.0 / T)
    resid = loc - gts[:, None]
    out = dict(f=np.fft.rfftfreq(T, d=1.0),                 # cycles/frame
               P_glob=_psd(gts), P_loc=_psd(resid).mean(axis=1),
               P_tot=_psd(loc).mean(axis=1),
               var_glob=float(gts.var()), var_loc=float(resid.var()), T=T)
    del loc, resid; gc.collect()
    return out

# ── D2 helper: per-pixel temporal kurtosis (Gaussian = 3) ──
def temporal_kurtosis(arr, chunk=200):
    T, H, W = arr.shape
    s1 = np.zeros((H, W), np.float64); s2 = np.zeros_like(s1)
    s3 = np.zeros_like(s1); s4 = np.zeros_like(s1)
    for s in range(0, T, chunk):
        b = arr[s:min(s + chunk, T)].astype(np.float64)
        s1 += b.sum(0); s2 += (b ** 2).sum(0); s3 += (b ** 3).sum(0); s4 += (b ** 4).sum(0)
    m1 = s1 / T; e2 = s2 / T; e3 = s3 / T; e4 = s4 / T
    var = e2 - m1 ** 2
    mu4 = e4 - 4 * m1 * e3 + 6 * m1 ** 2 * e2 - 3 * m1 ** 4
    return (mu4 / np.maximum(var, 1e-30) ** 2).astype(np.float32)

# ── shared: per-segment maps (same settings as the deliverable) ──
def maps_of(seg):
    d, _ = compute_density(seg, min_cv=0.005)
    g, _c, _z = compute_g_norm_torch(seg, RECON_TAUS, norm='nor1', min_cv=0.005)
    f = gpu_fit_maps(seg, recon_taus=RECON_TAUS, n_components=1, global_alpha=True,
                     gamma_scale=GAMMA_SCALE, min_cv=0.005, n_steps=400, verbose=False)
    try:    g = g.cpu().numpy()
    except Exception: g = np.asarray(g)
    return (f['gamma'].astype(np.float32), d.astype(np.float32),
            g.astype(np.float32), float(np.nanmedian(f['alpha'])))
print('loaders + diagnostic helpers defined')""")

# ─────────────────────────────── cell 4 ────────────────────────────────────
code(r"""# ── THE SINGLE LOAD: everything that needs the raw time series happens here ──
# Order is deliberate. D1/D2 need the movie, and the movie is freed at the end of
# this cell, so they CANNOT be moved into a later cell.
t0 = time.time()
raw = load_video(ZIP_PATH, VIDEO_FNAME, N_FRAMES, BIN_FACTOR)
T, H, W = raw.shape
nuc0 = load_mask((H, W))
print(f'nucleus px (mask only): {int(nuc0.sum())}  of {H*W}')

CACHE = {'T': T, 'H': H, 'W': W, 'nuc0': nuc0}

# D1a — spectrum of the RAW binned movie, BEFORE any flat-fielding.
# This matters: preprocess() divides each frame by its own smooth background, which
# removes much of the common mode. Instrumental lines are easiest to see here.
CACHE['psd_raw'] = psd_bundle(raw, nuc0, N_PSD_PX)
print(f'[D1a] raw PSD done  ({time.time()-t0:.0f}s)')

vp = preprocess(raw); del raw; gc.collect()

# D1b — spectrum of what actually reaches the analysis
CACHE['psd_pp'] = psd_bundle(vp, nuc0, N_PSD_PX)
print(f'[D1b] preprocessed PSD done  ({time.time()-t0:.0f}s)')

# D2 — per-pixel intensity distribution shape
CACHE['kurt'] = temporal_kurtosis(vp)
print(f'[D2] kurtosis done  ({time.time()-t0:.0f}s)')

# maps for the full record and both halves (same as the deliverable)
h = T // 2
gamma, dens, G, a_full = maps_of(vp)
g1g, g1d, G1, a_h1 = maps_of(vp[:h])
g2g, g2d, G2, a_h2 = maps_of(vp[h:])

# noise-free amplitude A = G(0+) extrapolated from tau>=1: 1-A is the white-noise share
try:
    CACHE['A_amp'] = vdls_amplitude(vp, RECON_TAUS, gamma,
                                    np.full_like(gamma, a_full))
except Exception as e:
    CACHE['A_amp'] = None; print('vdls_amplitude skipped:', e)

del vp; gc.collect()
try:
    import torch; torch.cuda.empty_cache()
except Exception: pass

nuc = nuc0 & np.isfinite(gamma)
CACHE.update(gamma=gamma, dens=dens, G=G, alpha_full=a_full,
             g1g=g1g, g2g=g2g, G1=G1, G2=G2, alpha_h1=a_h1, alpha_h2=a_h2,
             nuc=nuc)
print(f'\nloaded once, video freed.  nucleus px (analysed): {int(nuc.sum())}')
print(f'global alpha:  full={a_full:.3f}   half1={a_h1:.3f}   half2={a_h2:.3f}')
print(f'total {time.time()-t0:.0f}s')""")

# ─────────────────────────────── cell 5 : D1 ───────────────────────────────
code(r"""# ══ D1 — temporal power spectrum: are the dynamics sampled, or aliased? ══
@guard('D1')
def _d1():
    fs = FRAME_RATE_HZ or 1.0
    unit = 'Hz' if FRAME_RATE_HZ else 'cycles/frame'
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    summary = {}
    for ax, key, title in ((axes[0], 'psd_raw', 'RAW (before flat-field)'),
                           (axes[1], 'psd_pp', 'PREPROCESSED (what is analysed)')):
        b = CACHE[key]
        f = b['f'] * fs
        m = f > 0
        ax.loglog(f[m], b['P_glob'][m], lw=1.0, label='global (common mode)')
        ax.loglog(f[m], b['P_loc'][m], lw=1.0, label='local (per-pixel)')

        P = b['P_loc'][m]; ff = f[m]
        # white floor = median of the top decade; a genuine floor means the
        # dynamics have rolled off BEFORE Nyquist, i.e. no aliasing.
        top = ff > ff[-1] / 2.0
        floor = float(np.median(P[top]))
        f_nyq = ff[-1]
        white_frac = float(np.clip(floor * f_nyq / max(b['var_loc'], 1e-30), 0, 1))
        # slope over the top half-decade: ~0 = floor reached, clearly <0 = still falling
        sl_top = float(np.polyfit(np.log10(ff[top]), np.log10(np.maximum(P[top], 1e-30)), 1)[0])
        # mid-band power law -> an INDEPENDENT estimate of alpha (S ~ f^-(1+alpha))
        mid = (ff > ff[0] * 5) & (P > 3 * floor)
        if mid.sum() > 8:
            sl_mid = float(np.polyfit(np.log10(ff[mid]), np.log10(P[mid]), 1)[0])
            alpha_psd = -sl_mid - 1.0
        else:
            sl_mid, alpha_psd = float('nan'), float('nan')
        # sharp lines: peaks far above a running-median baseline
        base = median_filter(np.log10(np.maximum(P, 1e-30)), size=21)
        ratio = np.log10(np.maximum(P, 1e-30)) - base
        isline = ratio > np.log10(LINE_RATIO)
        lines = []
        if isline.any():
            idx = np.argsort(ratio)[::-1]
            taken = []
            for i in idx:
                if not isline[i]: break
                if any(abs(ff[i] - t) < ff[-1] * 0.01 for t in taken): continue
                taken.append(ff[i])
                gl = b['P_glob'][m][i] / max(P[i], 1e-30)
                lines.append((float(ff[i]), float(10 ** ratio[i]), float(gl)))
                if len(lines) >= 5: break
        ax.axhline(floor, color='k', ls=':', lw=1, label='white floor')
        for fl, _, _ in lines: ax.axvline(fl, color='r', ls='--', lw=0.7, alpha=0.6)
        ax.set_title(title, fontsize=10); ax.set_xlabel(f'frequency [{unit}]')
        ax.set_ylabel('PSD'); ax.legend(fontsize=7)
        summary[key] = dict(floor=floor, white_frac=white_frac, slope_top=sl_top,
                            slope_mid=sl_mid, alpha_psd=alpha_psd, lines=lines,
                            f_nyq=float(f_nyq))
    plt.tight_layout(); plt.show()

    s = summary['psd_pp']; sr = summary['psd_raw']
    print(f~~~
white-noise share of the per-pixel variance : {s['white_frac']*100:.1f}%  (preprocessed)
high-frequency slope (top half-decade)      : {s['slope_top']:+.2f}   (0 = floor reached)
mid-band power law S ~ f^b, b               : {s['slope_mid']:+.2f}
  -> alpha implied by the PSD (b = -(1+a))  : {s['alpha_psd']:.2f}
     alpha from the ACF fit                 : {CACHE['alpha_full']:.2f}   <- independent cross-check~~~)

    if s['lines'] or sr['lines']:
        print('\nsharp lines (freq, height above baseline, global/local power ratio):')
        for tag, dd in (('raw', sr), ('preprocessed', s)):
            for fl, ht, gl in dd['lines']:
                src = 'INSTRUMENT (common mode)' if gl > 3 else 'sample / mixed'
                print(f'  [{tag:12s}] {fl:9.4f} {unit}   x{ht:6.1f}   glob/loc={gl:7.1f}  {src}')
    else:
        print('\nno sharp lines above the threshold in either spectrum')

    # ---- verdicts ----
    aliasing = s['slope_top'] < -0.5
    v_alias = ('ALIASING RISK — the spectrum is still falling at Nyquist, so the fast '
               'dynamics are NOT resolved. No post-processing fixes this: raise the frame '
               'rate / shorten the exposure.') if aliasing else \
              ('OK — the spectrum flattens into a white floor before Nyquist, so the '
               'dynamics are sampled.')
    print('\nVERDICT D1-aliasing:', v_alias)
    inst = [l for l in (s['lines'] + sr['lines']) if l[2] > 3]
    print('VERDICT D1-lines   :', f'{len(inst)} instrumental line(s) found — a temporal '
          f'band-stop is worth doing (the spatial one in Section 3b does not touch these)'
          if inst else 'no common-mode instrumental line stands out')
    wf = s['white_frac'] * 100
    print('VERDICT D1-noise   :',
          f'white noise is {wf:.0f}% of the per-pixel variance; this is the measured '
          f'replacement for the 4.1% RMS assumed in docs/laplace_projection_alpha.md')

    # independent cross-check: the tau>=1 ACF is noise-free, so extrapolating its shape
    # back to tau->0 gives the noise-free amplitude A. vdls_amplitude fits C(tau)/<I>^2,
    # i.e. A is in CV^2 units -- the white share is 1 - A/CV^2, not 1 - A.
    wf_acf = float('nan')
    Amp = CACHE.get('A_amp')
    if Amp is not None:
        nucm = CACHE['nuc']
        ratio = (np.asarray(Amp, float) / np.maximum(CACHE['dens'], 1e-30))[nucm]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            wf_acf = float(1.0 - np.median(np.clip(ratio, 0.0, 1.0)))
            print(f'                     cross-check from the ACF extrapolation: '
                  f'{wf_acf*100:.0f}%  (independent of the PSD -- the two should agree; '
                  f'a big gap means one of the two estimators is being fooled)')
    RESULT['D1'] = dict(verdict='ok', aliasing=aliasing, white_frac=s['white_frac'],
                        white_frac_acf=wf_acf, n_inst_lines=len(inst),
                        alpha_psd=s['alpha_psd'], slope_top=s['slope_top'], summary=summary)""")

# ─────────────────────────────── cell 6 : D2 ───────────────────────────────
code(r"""# ══ D2 — intensity distribution shape: has the interference wrapped a fringe? ══
@guard('D2')
def _d2():
    k = CACHE['kurt']; nuc = CACHE['nuc']
    kk = k[nuc]; kk = kk[np.isfinite(kk)]
    frac_low = float((kk < KURT_LOW).mean())
    med = float(np.median(kk))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    kd = np.where(nuc, k, np.nan)
    im = ax[0].imshow(kd, cmap='coolwarm', vmin=np.nanpercentile(kd, 2),
                      vmax=np.nanpercentile(kd, 98))
    ax[0].set_title('per-pixel kurtosis (3 = Gaussian)', fontsize=10); ax[0].axis('off')
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(kk, bins=120, color='steelblue')
    ax[1].axvline(3.0, color='k', ls='--', label='Gaussian (linear regime)')
    ax[1].axvline(1.5, color='r', ls='--', label='full fringe wrap (arcsine)')
    ax[1].axvline(KURT_LOW, color='orange', ls=':', label=f'flag < {KURT_LOW}')
    ax[1].set_xlabel('kurtosis'); ax[1].legend(fontsize=7); ax[1].set_title('distribution', fontsize=10)
    gm = np.where(nuc, CACHE['gamma'], np.nan)
    ok = np.isfinite(gm) & np.isfinite(kd)
    ax[2].hexbin(np.log10(np.clip(gm[ok], 1e-6, None)), kd[ok], gridsize=60, cmap='viridis',
                 bins='log')
    ax[2].axhline(3.0, color='k', ls='--'); ax[2].set_xlabel('log10 gamma')
    ax[2].set_ylabel('kurtosis'); ax[2].set_title('kurtosis vs dynamics', fontsize=10)
    plt.tight_layout(); plt.show()

    print(f'median kurtosis in the nucleus : {med:.2f}   (3.0 = Gaussian)')
    print(f'fraction of pixels < {KURT_LOW}       : {frac_low*100:.1f}%')
    if med < KURT_LOW:
        v = ('WRAPPED — the bulk of the nucleus is platykurtic, i.e. the interference is '
             'bounded/oscillatory rather than linear in z. G(tau) is then NOT the ACF of a '
             'density-like quantity and the fitted (gamma, alpha) have no simple meaning.')
    elif frac_low > 0.10:
        v = (f'PARTIAL — {frac_low*100:.0f}% of pixels look wrapped. Those regions need to be '
             'masked out (or analysed as phase) before their gamma/alpha are trusted.')
    else:
        v = ('LINEAR REGIME — the intensity is close to Gaussian nearly everywhere, so the '
             'small-phase approximation behind the current model holds.')
    print('\nVERDICT D2:', v)
    RESULT['D2'] = dict(verdict='ok', median_kurt=med, frac_low=frac_low)""")

# ─────────────────────────────── cell 7 : D3 ───────────────────────────────
code(r"""# ══ D3 — model form: algebraic (FCS, q-integrated) vs stretched exp (single q_z)? ══
# If the decorrelation is dominated by AXIAL motion through the interference fringes,
# the correct form is the intermediate scattering function at a single, KNOWN q_z:
#     G(tau) = A exp(-q_z^2 D_z tau^alpha)        (stretched exponential)
# rather than the FCS-type algebraic form the pipeline currently fits:
#     G(tau) = A / (1 + gamma tau^alpha)
# Both have two shape parameters plus a free amplitude, so the comparison is fair.
@guard('D3')
def _d3():
    taus = np.asarray(RECON_TAUS, float)
    nuc = CACHE['nuc']
    def roi(stack):
        rows = np.asarray(stack, float)[nuc]
        return np.nanmean(rows[np.isfinite(rows).all(axis=1)], axis=0)
    g, g1, g2 = roi(CACHE['G']), roi(CACHE['G1']), roi(CACHE['G2'])
    noise = float(np.sqrt(((g1 - g2) ** 2).mean()) / 2.0)      # half-split, assumption-free

    def fit_shape(gc, kind, n_g=240, n_a=101):
        gams = np.logspace(-4, 1, n_g)[:, None, None]
        alps = np.linspace(0.2, 2.0, n_a)[None, :, None]
        ta = taus[None, None, :] ** alps
        s = 1.0 / (1.0 + gams * ta) if kind == 'alg' else np.exp(-gams * ta)
        A = (s * gc).sum(-1) / np.maximum((s * s).sum(-1), 1e-30)
        r = np.sqrt(((A[..., None] * s - gc) ** 2).mean(-1))
        i, j = np.unravel_index(np.argmin(r), r.shape)
        return dict(rms=float(r[i, j]), gamma=float(gams[i, 0, 0]),
                    alpha=float(alps[0, j, 0]), A=float(A[i, j]))

    alg, stx = fit_shape(g, 'alg'), fit_shape(g, 'str')
    print(f'ROI-averaged over {int(nuc.sum())} px;  half-split noise on the curve = {noise:.5f}\n')
    for tag, d in (('algebraic  A/(1+g t^a)', alg), ('stretched  A exp(-g t^a)', stx)):
        print(f'  {tag}:  rms={d["rms"]:.5f}   gamma={d["gamma"]:.4f}  alpha={d["alpha"]:.3f}  A={d["A"]:.3f}')
    d_rms = alg['rms'] - stx['rms']

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].semilogx(taus, g, 'ko', label='data (ROI mean)')
    tt = np.logspace(0, np.log10(taus.max()), 200)
    ax[0].semilogx(tt, alg['A'] / (1 + alg['gamma'] * tt ** alg['alpha']), label='algebraic')
    ax[0].semilogx(tt, stx['A'] * np.exp(-stx['gamma'] * tt ** stx['alpha']), label='stretched')
    ax[0].set_xlabel('tau [frames]'); ax[0].set_ylabel('G(tau)'); ax[0].legend(fontsize=8)
    ax[0].set_title('both fits, free amplitude', fontsize=10)

    # ln(-ln(G/A)) vs ln(tau): a STRAIGHT line of slope alpha means stretched exponential;
    # the algebraic form bends over and flattens. Illustrative -- the decisive number is
    # the residual comparison above, because this plot depends on the assumed A.
    yy = g / stx['A']
    ok = (yy > 1e-6) & (yy < 0.999999)
    if ok.sum() >= 4:
        x, y = np.log(taus[ok]), np.log(-np.log(yy[ok]))
        p1 = np.polyfit(x, y, 1); p2 = np.polyfit(x, y, 2)
        ax[1].plot(x, y, 'ko'); ax[1].plot(x, np.polyval(p1, x), label=f'linear, slope={p1[0]:.2f}')
        ax[1].plot(x, np.polyval(p2, x), '--', label=f'quadratic, curv={p2[0]:+.3f}')
        ax[1].set_xlabel('ln tau'); ax[1].set_ylabel('ln(-ln(G/A))'); ax[1].legend(fontsize=8)
        ax[1].set_title('straight = stretched exp; bent = algebraic', fontsize=10)
        curv = float(p2[0]); slope = float(p1[0])
    else:
        curv, slope = float('nan'), float('nan')
    plt.tight_layout(); plt.show()

    if abs(d_rms) < noise:
        v = (f'INDISTINGUISHABLE — the two forms differ by {abs(d_rms):.5f}, below the '
             f'{noise:.5f} half-split noise. The current algebraic model is not wrong, but '
             'it is not established either; do not read physics into gamma.')
        winner = 'tie'
    elif d_rms > 0:
        v = ('STRETCHED EXPONENTIAL WINS — consistent with axial-phase-dominated iSCAT. '
             'This is the good outcome: alpha is readable from the slope and gamma carries '
             'physical units at a known q_z.')
        winner = 'stretched'
    else:
        v = ('ALGEBRAIC WINS — decorrelation is dominated by lateral, q-integrated motion, '
             'so the current model is right and the single-q shortcut is unavailable.')
        winner = 'algebraic'
    print('VERDICT D3:', v)

    if winner in ('stretched', 'tie'):
        # gamma = q_z^2 D_z, with tau in frames
        Dz_frame = stx['gamma'] / Q_Z ** 2
        msg = f'  if stretched: D_z = {Dz_frame:.3e} um^2 / frame^alpha  (alpha={stx["alpha"]:.2f})'
        if FRAME_RATE_HZ:
            msg += f'\n                 = {Dz_frame * FRAME_RATE_HZ ** stx["alpha"]:.3e} um^2 / s^alpha'
        else:
            msg += '\n                 (set FRAME_RATE_HZ to get it per second)'
        print(msg)
        print('  NOTE q_z is fixed by the wavelength, so this needs no calibration --')
        print('  it is a physical number comparable across cells and instruments.')

    RESULT['D3'] = dict(verdict='ok', winner=winner, rms_alg=alg['rms'], rms_str=stx['rms'],
                        noise=noise, alpha_alg=alg['alpha'], alpha_str=stx['alpha'],
                        lnln_slope=slope, lnln_curv=curv)""")

# ─────────────────────────────── cell 8 : D4 ───────────────────────────────
code(r"""# ══ D4 — finite-T bias, measured directly instead of simulated ══
# The record is already split: gamma is fitted on T=5000, the halves on T=2500. If the
# sample-mean ACF bias matters, the halves carry twice as much of it, so
#   (a) the shared alpha should be HIGHER on the halves (a depressed slow tail reads as
#       a faster, less anomalous decay), and
#   (b) the gamma difference should DEPEND on gamma -- negative at the slow end and
#       crossing over at the fast end -- rather than being a constant offset.
# A constant offset would be harmless (every downstream projection absorbs it).
@guard('D4')
def _d4():
    nuc = CACHE['nuc']
    gf = CACHE['gamma']; gh = 0.5 * (CACHE['g1g'] + CACHE['g2g'])
    ok = nuc & np.isfinite(gf) & np.isfinite(gh) & (gf > 0) & (gh > 0)
    x = np.log10(gf[ok]); d = np.log10(gf[ok]) - np.log10(gh[ok])

    a_f, a_1, a_2 = CACHE['alpha_full'], CACHE['alpha_h1'], CACHE['alpha_h2']
    d_alpha = 0.5 * (a_1 + a_2) - a_f
    print(f'shared alpha:  full(T={CACHE["T"]})={a_f:.4f}   '
          f'halves(T={CACHE["T"]//2})={a_1:.4f}, {a_2:.4f}   mean-diff={d_alpha:+.4f}')

    qs = np.percentile(x, np.linspace(0, 100, 9))
    cx, cy, ce = [], [], []
    for i in range(len(qs) - 1):
        m = (x >= qs[i]) & (x < qs[i + 1])
        if m.sum() > 20:
            cx.append(float(np.median(x[m]))); cy.append(float(np.median(d[m])))
            ce.append(float(np.std(d[m]) / np.sqrt(m.sum())))
    cx, cy, ce = np.array(cx), np.array(cy), np.array(ce)
    trend = float(np.polyfit(cx, cy, 1)[0]) if len(cx) > 2 else float('nan')
    spread = float(cy.max() - cy.min()) if len(cy) else float('nan')

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].hexbin(x, d, gridsize=60, cmap='viridis', bins='log')
    ax[0].errorbar(cx, cy, yerr=ce, fmt='o-', color='r', lw=1.5, label='binned median')
    ax[0].axhline(0, color='k', ls='--', lw=1)
    ax[0].set_xlabel('log10 gamma (full record)')
    ax[0].set_ylabel('log10 gamma_full - log10 gamma_halves')
    ax[0].legend(fontsize=8); ax[0].set_title(f'trend slope = {trend:+.3f}', fontsize=10)
    dm = np.full(nuc.shape, np.nan); dm[ok] = d
    im = ax[1].imshow(dm, cmap='coolwarm', vmin=np.nanpercentile(dm, 2),
                      vmax=np.nanpercentile(dm, 98))
    ax[1].set_title('full - halves, spatially', fontsize=10); ax[1].axis('off')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    plt.tight_layout(); plt.show()

    print(f'binned median difference spans {spread:.4f} decades across the gamma range')
    if not np.isfinite(trend):
        v = 'INCONCLUSIVE — not enough bins'
    elif abs(trend) < 0.02 and spread < 0.02:
        v = ('NEGLIGIBLE / UNIFORM — no gamma-dependent trend. The preprocessing has already '
             'removed the slow common modes that drive the bias, so section 6 of '
             'docs/laplace_projection_alpha.md does not apply to this data and D5 can be '
             'read at face value.')
    else:
        v = (f'PRESENT AND GAMMA-DEPENDENT (slope {trend:+.3f}) — this is the signal-dependent '
             'case, so it distorts contrast rather than shifting it, AND it reproduces across '
             'halves, so no reliability test in the pipeline can catch it. Correct it before '
             'trusting any slow-end conclusion (see section 6.4).')
    print('\nVERDICT D4:', v)
    RESULT['D4'] = dict(verdict='ok', trend=trend, spread=spread, d_alpha=d_alpha)""")

# ─────────────────────────────── cell 9 : D5 ───────────────────────────────
code(r"""# ══ D5 — bulk rate spectrum (only meaningful if D1-D4 came back clean) ══
# Asks a question the bulk (gamma, CV^2) histogram structurally cannot: not "are there
# two kinds of pixel?" (spatial) but "are there two decay rates inside one ACF?"
# (dynamical), which is what sub-pixel coexistence looks like.
@guard('D5')
def _d5():
    nuc = CACHE['nuc']
    Y = np.log10(np.clip(CACHE['dens'], 1e-12, None))
    nucleoli = nuc & (Y < np.nanpercentile(Y[nuc], 12))
    nucleoplasm = nuc & ~nucleoli

    for tag, m in (('whole nucleus', nuc), ('nucleoli', nucleoli), ('nucleoplasm', nucleoplasm)):
        if int(m.sum()) < 50:
            print(f'{tag}: only {int(m.sum())} px, skipped'); continue
        r = lsp.roi_spectrum(CACHE['G'], m, RECON_TAUS, G1=CACHE['G1'], G2=CACHE['G2'])
        print(f'\n--- {tag} ({r["n_px"]} px) ---')
        print(f'  peaks={r["n_peaks"]}  centroid={r["centroid"]:.2f}  width={r["width"]:.2f} dec'
              f'  half1/half2 peaks={r.get("n_peaks_half1")}/{r.get("n_peaks_half2")}')
        print(f'  1-comp reference: gamma={r["gamma_1comp"]:.4f} alpha={r["alpha_1comp"]:.2f}'
              f'  residual={r["resid_1comp"]:.5f}  vs noise {r["noise_rms"]:.5f}')
        print(f'  {r["verdict"]}')
        RESULT.setdefault('D5', {})[tag] = dict(n_px=r['n_px'], n_peaks=r['n_peaks'],
                                                width=r['width'], resid=r['resid_1comp'],
                                                noise=r['noise_rms'],
                                                bias_limited=r.get('bias_limited'))
        if tag == 'whole nucleus':
            plt.figure(figsize=(6, 3.6))
            plt.semilogx(r['rates'], r['p'] / max(r['p'].max(), 1e-30), label='full')
            if 'p_half1' in r:
                plt.semilogx(r['rates_halves'], r['p_half1'] / max(r['p_half1'].max(), 1e-30),
                             '--', lw=0.9, label='half 1')
                plt.semilogx(r['rates_halves'], r['p_half2'] / max(r['p_half2'].max(), 1e-30),
                             '--', lw=0.9, label='half 2')
            plt.xlabel('decay rate Gamma [1/frame]'); plt.ylabel('p (normalised)')
            plt.title('bulk rate spectrum — a peak that does not repeat is noise', fontsize=10)
            plt.legend(fontsize=8); plt.tight_layout(); plt.show()
    RESULT.setdefault('D5', {})['verdict'] = 'ok'""")

# ─────────────────────────────── cell 10 ───────────────────────────────────
code(r"""# ══ scorecard: measured outcome vs the predictions registered in cell 0 ══
def _fmt(ok):
    return {True: 'PREDICTED', False: 'SURPRISE ', None: '  n/a    '}[ok]

rows = []
d1, d2, d3, d4, d5 = (RESULT.get(k) or {} for k in ('D1', 'D2', 'D3', 'D4', 'D5'))

if d1.get('verdict') == 'ok':
    rows.append(('PSD has an instrumental line (~80%)', d1['n_inst_lines'] > 0,
                 f"{d1['n_inst_lines']} line(s)"))
    rows.append(('white noise 30-70% of variance (~70%)',
                 0.30 <= d1['white_frac'] <= 0.70, f"{d1['white_frac']*100:.0f}%"))
    rows.append(('no aliasing (~60%)', not d1['aliasing'],
                 f"top-band slope {d1['slope_top']:+.2f}"))
if d2.get('verdict') == 'ok':
    rows.append(('kurtosis mostly ~3 with a low tail (~55%)',
                 d2['median_kurt'] > KURT_LOW, f"median {d2['median_kurt']:.2f}, "
                 f"{d2['frac_low']*100:.0f}% below {KURT_LOW}"))
if d3.get('verdict') == 'ok':
    rows.append(('stretched exp at least as good (~55%)', d3['winner'] in ('stretched', 'tie'),
                 f"winner: {d3['winner']}  (alg {d3['rms_alg']:.5f} vs str {d3['rms_str']:.5f}, "
                 f"noise {d3['noise']:.5f})"))
if d4.get('verdict') == 'ok':
    small = abs(d4['trend']) < 0.02 and d4['spread'] < 0.02
    rows.append(('finite-T bias smaller than simulated (~65%)', small,
                 f"trend {d4['trend']:+.3f}, span {d4['spread']:.3f} dec, "
                 f"d_alpha {d4['d_alpha']:+.3f}"))
wn = d5.get('whole nucleus')
if wn:
    rows.append(('bulk spectrum single-peaked (~75%)', wn['n_peaks'] <= 1,
                 f"{wn['n_peaks']} peak(s), width {wn['width']:.2f} dec"))

print('=' * 96)
print('PREDICTION SCORECARD'.center(96))
print('=' * 96)
for name, ok, detail in rows:
    print(f'  {_fmt(ok)} | {name:44s} | {detail}')
print('=' * 96)
hit = sum(1 for _, ok, _ in rows if ok)
print(f'  {hit}/{len(rows)} predictions held.')
print('  Surprises are the useful rows — they are where the model of the experiment was wrong.')

# what to do next, driven by what actually came back
print('\nNEXT STEP:')
if d1.get('aliasing'):
    print('  * D1 says aliased. Nothing downstream is trustworthy. Re-acquire faster;')
    print('    no analysis change can recover unsampled dynamics.')
elif d2.get('median_kurt', 3) < KURT_LOW:
    print('  * D2 says the interference has wrapped. Treat the signal as phase, not')
    print('    intensity: unwrap, or restrict to the pixels that are still Gaussian.')
elif d3.get('winner') == 'stretched':
    print('  * D3 says single-q axial physics. Switch the fitter to the stretched form,')
    print('    read alpha off the slope, and report D_z in um^2/s^alpha. This also gives a')
    print('    real cross-cell comparison, which is what the slope-3 question needs.')
elif d4.get('trend') is not None and abs(d4.get('trend', 0)) >= 0.02:
    print('  * D4 says the finite-T bias is real and gamma-dependent. Correct it')
    print('    (docs/laplace_projection_alpha.md section 6.4) before anything else.')
else:
    print('  * Nothing invalidating found. The measurement is what it was assumed to be,')
    print('    and the ~1-robust-axis conclusion stands. New axes need new acquisition')
    print('    (q-resolved DDM/STICS, or an orthogonal modality).')""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "/home/user/iscors-net/iscors_one_shot_diagnostics.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "cells:", len(CELLS))
