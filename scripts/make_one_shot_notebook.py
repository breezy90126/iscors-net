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
| **D6** | 用模擬 iPSF kernel 反演 P(D)（µm²/s） | 修正 D5 的 kernel mismatch；見 `docs/ipsf_kernel.md` |

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
BRANCH = 'claude/project-thread-kkiy52'
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
import utils.ipsf_kernel as ik
import utils.finite_t as ft

RESULT = {}          # every diagnostic writes its verdict here; last cell tabulates
def guard(name):
    ~~~Decorator: a failing diagnostic must not abort the one-shot run.~~~
    def deco(fn):
        _CUR[0] = name
        try:
            fn()
        except Exception:
            RESULT[name] = {'verdict': 'ERROR', 'detail': traceback.format_exc(limit=3)}
            print(f'!! {name} failed (run continues):'); traceback.print_exc(limit=3)
    return deco

# ── save every figure + all printed text to Drive (OUT_DIR is set in the config cell) ──
import builtins, json as _json, datetime as _dt
_FIG_N = [0]; _CUR = ['setup']
def _out_dir():
    d = globals().get('OUT_DIR') or '/content/drive/MyDrive/iscors_test/one_shot_outputs'
    os.makedirs(d, exist_ok=True); return d
_show = plt.show
def _save_show(*a, **kw):
    fig = plt.gcf()
    if fig.get_axes():
        _FIG_N[0] += 1
        fn = os.path.join(_out_dir(), f'fig{_FIG_N[0]:02d}_{_CUR[0]}.png')
        fig.savefig(fn, dpi=150, bbox_inches='tight'); builtins.print(f'  [saved] {fn}')
    return _show(*a, **kw)
plt.show = _save_show
_print = builtins.print
def print(*a, **kw):                     # tee: screen + OUT_DIR/log.txt
    _print(*a, **kw)
    if kw.get('file') is None:
        with open(os.path.join(_out_dir(), 'log.txt'), 'a') as f:
            _print(*a, **{k: v for k, v in kw.items() if k != 'file'}, file=f)
def save_results():
    ~~~RESULT (all verdicts / numbers) -> OUT_DIR/results.json~~~
    fn = os.path.join(_out_dir(), 'results.json')
    with open(fn, 'w') as f:
        _json.dump(RESULT, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    _print('[saved]', fn)
print('imports ok')""")

# ─────────────────────────────── cell 2 ────────────────────────────────────
code(r"""# ── config (EDIT paths + the acquisition block) ──
# Data: the paper's Zenodo record (Hsiao et al., Commun. Biol. 7:763, 2024),
# iSCORS_map_analysis.zip - the next cell downloads it if it is not on Drive yet.
ZIP_PATH    = '/content/drive/MyDrive/iscors_test/large_file.zip'   # .zip or a .tif
VIDEO_FNAME = 'COBRI_rarw_video.tif'
EXTRACT_DIR = '/content/real_data'
MASK_PATH   = 'data/condensation_mask.tif'
N_FRAMES = 5000; BIN_FACTOR = 2; RECON_TAUS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
GAMMA_SCALE = 2.0

# ── acquisition / optics (EDIT — D1 and D3 read these) ──
FRAME_RATE_HZ = 5000.0    # paper: 5000 fps unless stated (1000 fps for long time-lapses). None -> cycles/frame
WAVELENGTH_NM = 660.0     # paper Methods: 660 nm laser (Laser Quantum opus 660)
N_MEDIUM      = 1.38      # refractive index of the nucleoplasm

# ── D6: iPSF kernel (paper Methods: COBRI = iSCAT in transmission, OBJ2 NA 1.49,
#        640x640 camera at 42 nm/px). The kernel needs the RAW pixel; BIN_FACTOR is applied on top.
CAM_PIXEL_NM  = 42.0
OBJ_NA        = 1.49
GEOMETRY      = 'transmission'
IPSF_NPZ      = None      # path from scripts/export_ipsf_stack.py; None -> scalar pupil model
IPSF_Z_NM     = (-300, -150, 0, 150, 300)   # static height spread averaged into the kernel
D6_LAMBDAS    = (1e-4, 1e-3, 1e-2)   # CONTIN regularisation scan: only peaks that survive all are real
D6_REL_MIN    = 0.30      # GEVD axes kept if their cross-half reproducible fraction >= this
D6_DROP_FRAC  = 0.30      # drop this share of nucleus pixels with the worst half1-vs-half2 agreement
D6_N_CLUSTERS = 3         # k-means groups in the reproducible-axis space (P(D) per group)
D6_FEATURES = 'G+CV2'     # noise-filter features: 'G+CV2' (Section 3), 'XY' (tau map + CV2, Section 2), 'G'
HP_W = 1501               # D9 slow-peak test: temporal high-pass (boxcar, frames); 0 = skip
TILE_PX = 12              # D10 tile CONTIN map: tile size in binned px

# ── outputs: every figure (png), the printed log and results.json land here ──
OUT_DIR = '/content/drive/MyDrive/iscors_test/one_shot_outputs/' + time.strftime('%Y%m%d_%H%M')

# ── diagnostic knobs ──
N_PSD_PX   = 2000         # pixels sampled for the local power spectrum
LINE_RATIO = 5.0          # a PSD peak this far above the local baseline counts as a "line"
KURT_LOW   = 2.5          # below this, the interference has likely wrapped through a fringe

Q_Z = 4 * np.pi * N_MEDIUM / (WAVELENGTH_NM * 1e-3)     # rad/um, axial interference wavevector
print(f'config ready.  q_z = {Q_Z:.1f} rad/um  ->  one fringe every '
      f'{2 * np.pi / Q_Z * 1000:.0f} nm of axial motion')
print('frame rate:', FRAME_RATE_HZ or '(unset -> cycles/frame)')""")

# ─────────────────────────────── cell 2b : data download ───────────────────
code(r"""# ── data download (skip if the zip is already on Drive) ──
# Direct link to the paper's Zenodo data (Hsiao et al. 2024, record 11182995).
# Downloads straight into Colab (fast), then copies to Drive so later runs skip this.
import shutil
DATA_URL       = 'https://zenodo.org/records/11182995/files/iSCORS_map_analysis.zip?download=1'
DRIVE_DATA_DIR = '/content/drive/MyDrive/iscors_test'
DRIVE_ZIP      = os.path.join(DRIVE_DATA_DIR, 'iSCORS_map_analysis.zip')
LOCAL_ZIP      = '/content/iSCORS_map_analysis.zip'

if os.path.exists(DRIVE_ZIP):
    print('already on Drive:', DRIVE_ZIP)
    ZIP_PATH = DRIVE_ZIP
elif os.path.exists(ZIP_PATH):
    print('using existing ZIP_PATH:', ZIP_PATH)
else:
    if not os.path.exists(LOCAL_ZIP):
        subprocess.run(['wget', '-q', '--show-progress', '-O', LOCAL_ZIP, DATA_URL], check=True)
    os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
    shutil.copy(LOCAL_ZIP, DRIVE_ZIP); print('copied to Drive:', DRIVE_ZIP)
    ZIP_PATH = LOCAL_ZIP                     # read the local copy this session (faster)

# list what is inside, so VIDEO_FNAME can be checked before the long load
with zipfile.ZipFile(ZIP_PATH) as z:
    infos = sorted(z.infolist(), key=lambda x: -x.file_size)
print('zip contents (largest first):')
for x in infos[:15]:
    print(f'  {x.file_size/1e9:7.2f} GB  {x.filename}')
names = [os.path.basename(x.filename) for x in infos]
if VIDEO_FNAME not in names:
    tifs = [os.path.basename(x.filename) for x in infos if x.filename.lower().endswith(('.tif', '.tiff'))]
    if tifs:
        print(f'VIDEO_FNAME {VIDEO_FNAME!r} not in zip -> using the largest tif {tifs[0]!r}')
        VIDEO_FNAME = tifs[0]
    else:
        print('!! no .tif in the zip - set VIDEO_FNAME / ZIP_PATH by hand')
print('ZIP_PATH =', ZIP_PATH, '   VIDEO_FNAME =', VIDEO_FNAME)""")

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
    g, _c, c0 = compute_g_norm_torch(seg, RECON_TAUS, norm='nor1', min_cv=0.005)
    f = gpu_fit_maps(seg, recon_taus=RECON_TAUS, n_components=1, global_alpha=True,
                     gamma_scale=GAMMA_SCALE, min_cv=0.005, n_steps=400, verbose=False)
    try:    g, c0 = g.cpu().numpy(), c0.cpu().numpy()
    except Exception: g, c0 = np.asarray(g), np.asarray(c0)
    return (f['gamma'].astype(np.float32), d.astype(np.float32),
            g.astype(np.float32), float(np.nanmedian(f['alpha'])), c0.astype(np.float32))
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
gamma, dens, G, a_full, C0 = maps_of(vp)
g1g, g1d, G1, a_h1, C0_1 = maps_of(vp[:h])
g2g, g2d, G2, a_h2, C0_2 = maps_of(vp[h:])

# D9 input: the same movie high-passed in time (centred HP_W-frame boxcar, in place) and
# trimmed by HP_W//2 at both ends so every frame had a full window. Removes per-pixel drift
# slower than ~HP_W frames; the D9 kernel applies the identical filter, so a stationary slow
# population should come back unchanged and only non-stationary drift should disappear.
if HP_W:
    try:
        vp = np.asarray(vp, np.float32)
        ik.highpass_video_(vp, HP_W)
        _m = HP_W // 2; vh = vp[_m:T - _m]; Th = vh.shape[0]; hh = Th // 2
        def _gn(seg):
            g_, _c, c0_ = compute_g_norm_torch(seg, RECON_TAUS, norm='nor1', min_cv=0.0)
            try: g_, c0_ = g_.cpu().numpy(), c0_.cpu().numpy()
            except Exception: g_, c0_ = np.asarray(g_), np.asarray(c0_)
            return g_.astype(np.float32), c0_.astype(np.float32)
        G_hp, C0_hp = _gn(vh); G1_hp, C0_1_hp = _gn(vh[:hh]); G2_hp, C0_2_hp = _gn(vh[hh:])
        CACHE.update(G_hp=G_hp, G1_hp=G1_hp, G2_hp=G2_hp, C0_hp=C0_hp, C0_1_hp=C0_1_hp,
                     C0_2_hp=C0_2_hp, T_hp=Th)
        del vh
        print(f'[D9] high-passed G done (W={HP_W}, T_eff={Th})  ({time.time()-t0:.0f}s)')
    except Exception as e:
        print('[D9] high-pass skipped:', e)

del vp; gc.collect()
try:
    import torch; torch.cuda.empty_cache()
except Exception: pass

nuc = nuc0 & np.isfinite(gamma)
CACHE.update(C0=C0, C0_1=C0_1, C0_2=C0_2)
CACHE.update(g1d=g1d, g2d=g2d)
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
        # PSD is normalised per cycles/frame (sum P df = var, df = 1/T), so the white
        # variance is floor x Nyquist IN CYCLES/FRAME (0.5), not in Hz. Using f in Hz here
        # inflated the share by FRAME_RATE_HZ and clipped it to 100% (fixed).
        f_nyq_cpf = float(b['f'][m][-1])
        white_frac = float(np.clip(floor * f_nyq_cpf / max(b['var_loc'], 1e-30), 0, 1))
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

    # independent cross-check: the tau>=1 ACF is noise-free, so a FREE-amplitude fit
    # A/(1+gamma tau^alpha) (finite-T aware, utils/finite_t.py) on the nor1 curves gives the
    # white share as 1 - A. (The old cross-check extrapolated with the pipeline's gamma, which
    # was fitted WITHOUT an amplitude, so it absorbed the white-noise drop and read ~1%.)
    wf_acf = float('nan')
    try:
        nucm = CACHE['nuc'] & np.isfinite(CACHE['G']).all(-1)
        Gs = CACHE['G'][nucm].astype(np.float64)
        rs_ = np.random.RandomState(0)
        Gs = Gs[rs_.choice(len(Gs), size=min(5000, len(Gs)), replace=False)]
        fx = ft.debias(Gs, np.asarray(RECON_TAUS), CACHE['T'])
        wf_acf = float(1.0 - np.median(fx['A']))
        print(f'                     cross-check from a free-amplitude ACF fit (5000 px): '
              f'{wf_acf*100:.0f}%  (independent of the PSD -- the two should agree; '
              f'a big gap means one of the two estimators is being fooled)')
    except Exception as e:
        print('ACF cross-check skipped:', e)
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
# ── D4b — finite-T correction (docs/laplace_projection_alpha.md 6.4 item 3, exact form) ──
# Fit the EXPECTED pipeline estimator (sample-mean subtraction over T frames, / C_hat(0))
# instead of the ideal ACF: utils/finite_t.py, same formula as the D6 kernel. The full record
# is fitted with T, each half with T/2. If finite-T is what D4 saw, the corrected full and
# half gammas agree (no gamma-dependent trend) and the shared alpha agrees.
# Two levels:
#   per pixel  : noisy; the fit itself is biased at this noise, so expect only partial agreement
#   gamma bins : 8 c0-weighted ROI curves (pixels binned by gamma_full) -- near noise-free,
#                this is the decisive test, and it is the level D5/D6 work at.
# The corrected curves (g - bias) replace G/G1/G2 in D5 as a second, 'corrected' run.
def _d4_correct(ok, gf):
    T = CACHE['T']; taus = np.asarray(RECON_TAUS)
    G, G1, G2 = (CACHE[k][ok].astype(np.float64) for k in ('G', 'G1', 'G2'))
    good = np.isfinite(G).all(1) & np.isfinite(G1).all(1) & np.isfinite(G2).all(1)
    t0 = time.time()
    rf = ft.debias(G[good], taus, T)
    r1 = ft.debias(G1[good], taus, T // 2)
    r2 = ft.debias(G2[good], taus, T // 2)
    print(f'\n[D4b] finite-T corrected per-pixel fits done ({time.time()-t0:.0f}s, '
          f'{int(good.sum())} px)')

    def _summ(key_g, key_a, tag):
        gh_ = np.sqrt(r1[key_g] * r2[key_g])
        x = np.log10(rf[key_g]); d = x - np.log10(gh_)
        qs = np.percentile(x, np.linspace(0, 100, 9))
        cy = [np.median(d[(x >= qs[i]) & (x < qs[i + 1])]) for i in range(8)]
        tr = float(np.polyfit(x, d, 1)[0])
        da = 0.5 * (r1[key_a][0] + r2[key_a][0]) - rf[key_a][0]
        print(f'  per pixel, {tag:9s}: alpha full={rf[key_a][0]:.2f} halves={r1[key_a][0]:.2f},'
              f'{r2[key_a][0]:.2f} (d={da:+.2f})  trend={tr:+.3f}  '
              f'binned span={max(cy)-min(cy):.3f} dec')
        return tr, float(max(cy) - min(cy)), float(da)
    tr_n, sp_n, da_n = _summ('gamma_raw', 'alpha_raw', 'naive')
    tr_c, sp_c, da_c = _summ('gamma', 'alpha', 'corrected')

    # gamma-binned ROI curves (ratio of means, c0-weighted) -- the decisive level
    C0s = [CACHE[k][ok][good].astype(np.float64) for k in ('C0', 'C0_1', 'C0_2')]
    xb = np.log10(rf['gamma_raw'])
    qs = np.percentile(xb, np.linspace(0, 100, 9))
    lab = np.clip(np.digitize(xb, qs[1:-1]), 0, 7)
    def _bins(Gs, c0):
        return np.array([(Gs[lab == i] * c0[lab == i, None]).sum(0) /
                         max(c0[lab == i].sum(), 1e-30) for i in range(8)])
    Bf = _bins(G[good], C0s[0]); B1 = _bins(G1[good], C0s[1]); B2 = _bins(G2[good], C0s[2])
    bf = ft.debias(Bf, taus, T); b1 = ft.debias(B1, taus, T // 2); b2 = ft.debias(B2, taus, T // 2)
    res = {}
    for tag, kg, ka in (('naive', 'gamma_raw', 'alpha_raw'), ('corrected', 'gamma', 'alpha')):
        d = np.log10(bf[kg]) - 0.5 * (np.log10(b1[kg]) + np.log10(b2[kg]))
        da = 0.5 * (b1[ka][0] + b2[ka][0]) - bf[ka][0]
        res[tag] = dict(d=d, span=float(d.max() - d.min()), d_alpha=float(da),
                        span_inner=float(d[1:].max() - d[1:].min()),
                        alpha=float(bf[ka][0]))
        print(f'  gamma bins, {tag:9s}: alpha full={bf[ka][0]:.2f} halves={b1[ka][0]:.2f},'
              f'{b2[ka][0]:.2f} (d={da:+.2f})  dlog10 gamma per bin = {np.round(d, 3)}')
    A_bins = bf['A']
    print(f'  corrected white-noise share per gamma bin (1-A): {np.round(1 - A_bins, 2)}  '
          f'<- second cross-check for D1')

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    xc = [np.median(xb[lab == i]) for i in range(8)]
    for tag, mk in (('naive', 'o-'), ('corrected', 's-')):
        ax[0].plot(xc, res[tag]['d'], mk, label=f"{tag}  (alpha {res[tag]['alpha']:.2f}, "
                                                 f"d_alpha {res[tag]['d_alpha']:+.2f})")
    ax[0].axhline(0, color='k', ls='--', lw=1)
    ax[0].set_xlabel('log10 gamma bin (naive, full record)')
    ax[0].set_ylabel('log10 gamma_full - log10 gamma_halves')
    ax[0].set_title('gamma-binned ROI curves: finite-T correction', fontsize=10)
    ax[0].legend(fontsize=8)
    for i in (0, 3, 7):
        ax[1].semilogx(taus, Bf[i], 'o', ms=3, color=f'C{i}')
        ax[1].semilogx(taus, bf['g_corr'][i], '-', color=f'C{i}', label=f'bin {i}: corrected')
    ax[1].set_xlabel('tau [frames]'); ax[1].set_ylabel('G (nor1)')
    ax[1].set_title('measured (dots) vs finite-T corrected (lines)', fontsize=10)
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.show()

    n, c = res['naive'], res['corrected']
    # what matters is the GAMMA-DEPENDENCE (span), not a constant offset or a small alpha shift:
    # in a pure finite-T simulation the corrected bins keep a ~-0.05 dec constant offset and
    # |d_alpha| ~ 0.04-0.06 (gamma mixing inside a bin), so those are not evidence of drift.
    fixed = c['span_inner'] < 0.5 * n['span_inner']
    if fixed:
        v = ('FINITE-T EXPLAINS IT — after correction the full and half records agree '
             f"(span {n['span_inner']:.3f} -> {c['span_inner']:.3f} dec, excluding the slowest bin; "
             f"d_alpha {n['d_alpha']:+.3f} -> {c['d_alpha']:+.3f}). D5 is re-run on the corrected curves; "
             'D6 already has the bias inside its kernel.')
    else:
        v = ('FINITE-T DOES NOT EXPLAIN ALL OF IT — the correction leaves a gamma-dependent '
             f"difference (span {n['span_inner']:.3f} -> {c['span_inner']:.3f} dec; d_alpha "
             f"{n['d_alpha']:+.3f} -> {c['d_alpha']:+.3f}). The rest is something the stationary "
             'model lacks (slow drift / non-stationarity / bleaching): the slow end, and the '
             'slow D6 peak, stay unconfirmed.')
    print('\nVERDICT D4b:', v)

    for key, r in (('G_ft', rf), ('G1_ft', r1), ('G2_ft', r2)):
        arr = np.full(CACHE['G'].shape, np.nan, np.float32)
        idx = np.flatnonzero(ok)[good]
        arr.reshape(-1, arr.shape[-1])[idx] = r['g_corr'].astype(np.float32)
        CACHE[key] = arr
    RESULT['D4b'] = dict(verdict='ok', fixed=bool(fixed),
                         span_naive=n['span_inner'], span_corr=c['span_inner'],
                         d_alpha_naive=n['d_alpha'], d_alpha_corr=c['d_alpha'],
                         alpha_corr=c['alpha'], white_share_bins=(1 - A_bins).tolist(),
                         pix_trend_naive=tr_n, pix_trend_corr=tr_c)


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
    RESULT['D4'] = dict(verdict='ok', trend=trend, spread=spread, d_alpha=d_alpha)
    _d4_correct(ok, gf)
""")

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

    variants = [('', 'G', 'G1', 'G2')]
    if 'G_ft' in CACHE:
        variants.append((' [finite-T corrected]', 'G_ft', 'G1_ft', 'G2_ft'))
    for suffix, kG, k1, k2 in variants:
      for tag0, m in (('whole nucleus', nuc), ('nucleoli', nucleoli), ('nucleoplasm', nucleoplasm)):
        tag = tag0 + suffix
        if int(m.sum()) < 50:
            print(f'{tag}: only {int(m.sum())} px, skipped'); continue
        mm = m & np.isfinite(CACHE[kG]).all(-1)
        r = lsp.roi_spectrum(CACHE[kG], mm, RECON_TAUS, G1=CACHE[k1], G2=CACHE[k2],
                             bias_corrected=kG.endswith('_ft'))
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

# ─────────────────────────────── cell 9b : D6 ──────────────────────────────
code(r"""# ══ D6 — P(D) with the iPSF kernel, on GEVD-filtered pixels (docs/ipsf_kernel.md) ══
# 1) noise filter, same idea as the deliverable's Section 3 (pca-separation line):
#    cross-half GEVD on the per-pixel G(tau) curves. An axis's eigenvalue is the share of its
#    variance that REPEATS between the two halves; axes below D6_REL_MIN are noise. Each pixel
#    is scored by how well its half1 and half2 projections on the kept axes agree; the worst
#    D6_DROP_FRAC are dropped. The survivors are k-means clustered on the kept axes.
# 2) per group: c0-weighted ROI curve -> iPSF kernel (finite-T bias for full / half length)
#    -> CONTIN with a separate white-noise weight, scanned over D6_LAMBDAS.
# Reading rules (printed as VERDICT):
#    * mass at the grid edges is NOT a population: slow edge = still correlated at tau_max
#      (slower than this window resolves), fast edge = decays within ~1 frame.
#    * a multi-peak claim needs (a) residual within ~2x the half-split noise and (b) the same
#      interior peaks at every lambda. Half-split agreement alone is not enough: a systematic
#      (finite-T bias, drift - see D4) is identical in both halves.
def _d6_kernel():
    ~~~iPSF spectrum + finite-T kernels for full and half records (cached).~~~
    if 'd6_kern' in CACHE:
        return CACHE['d6_kern']
    assert FRAME_RATE_HZ, 'D6 needs FRAME_RATE_HZ to put D in physical units'
    dt = 1.0 / FRAME_RATE_HZ
    px = CAM_PIXEL_NM * 1e-9
    if IPSF_NPZ:
        stack, zs, px = ik.load_ipsf_npz(IPSF_NPZ); src = f'generator stack {IPSF_NPZ}'
    else:
        stack = ik.scalar_ipsf_stack([z * 1e-9 for z in IPSF_Z_NM], n_pix=256, pixel_size=px,
                                     wavelength=WAVELENGTH_NM * 1e-9, NA=OBJ_NA,
                                     n_medium=N_MEDIUM, geometry=GEOMETRY)
        src = 'scalar pupil model (set IPSF_NPZ for the vector generator)'
    q2, S = ik.ipsf_spectrum(stack, px, bin_factor=BIN_FACTOR, n_q2_bins=512)
    q2e = ik.effective_q2(q2, S)
    print(f'iPSF source: {src}')
    print(f'effective q = {np.sqrt(q2e)*1e-6:.2f} rad/um  (1/q = {1e9/np.sqrt(q2e):.0f} nm)'
          '  <- the absolute D scale goes as 1/q^2: a 2x error in q is 4x in D')
    T, h = CACHE['T'], CACHE['T'] // 2
    D_grid = ik.d_grid(RECON_TAUS, dt, q2e, n=80, pad=10.0)
    K_full, _ = ik.ipsf_kernel(RECON_TAUS, dt, q2, S, D=D_grid, T=T)
    K_half, _ = ik.ipsf_kernel(RECON_TAUS, dt, q2, S, D=D_grid, T=h)
    D_um = D_grid * 1e12
    print(f'resolvable window: D ~ {D_um[3]:.3g} .. {D_um[-4]:.3g} um^2/s')
    CACHE['d6_kern'] = (K_full, K_half, D_um)
    CACHE['d6_q2S'] = (q2, S, D_grid, dt)
    return CACHE['d6_kern']


def _pd_groups(groups, title, keys=('G', 'G1', 'G2', 'C0', 'C0_1', 'C0_2'), kern=None):
    ~~~CONTIN P(D) for each (tag, mask) ROI; prints verdicts, plots one panel per ROI.~~~
    K_full, K_half, D_um = kern or _d6_kernel()
    G, G1, G2 = (CACHE[k] for k in keys[:3])
    C0k, C01k, C02k = keys[3:]
    out = {}
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    for (tag, m), axp in zip(groups, axs):
        g, n_px = ik.roi_curve(G, CACHE[C0k], m)
        g1, _ = ik.roi_curve(G1, CACHE[C01k], m)
        g2, _ = ik.roi_curve(G2, CACHE[C02k], m)
        if n_px < 50:
            print(f'{tag}: only {n_px} px, skipped'); axp.set_title(f'{tag}: {n_px} px, skipped', fontsize=9)
            continue
        noise = float(np.sqrt(((g1 - g2) ** 2).mean()) / 2.0)
        D1, r1 = ik.best_single_d(g, K_full, D_um)
        print(f'\n--- {tag} ({n_px} px) ---   G(1)={g[0]:.3f}  G(tau_max)={g[-1]:.3f}  noise {noise:.5f}')
        print(f'  single-D reference: D = {D1:.3g} um^2/s, residual {r1:.5f}')
        runs = []
        for lam in D6_LAMBDAS:
            p, nug = ik.invert_kernel(g, K_full, lam=lam, return_nugget=True)
            p1 = ik.invert_kernel(g1, K_half, lam=lam); p2 = ik.invert_kernel(g2, K_half, lam=lam)
            sm, s1, s2 = (ik.spectrum_summary(x, D_um) for x in (p, p1, p2))
            resid = float(np.sqrt(((K_full @ p - g) ** 2).mean()))
            runs.append(dict(lam=lam, p=p, nug=nug, s=sm, resid=resid,
                             halves_same=len(s1['peak_D']) == len(s2['peak_D']) == len(sm['peak_D'])))
            print(f'  lam={lam:.0e}: white-noise {nug:.2f} | slow-edge {sm["slow_edge_frac"]:.2f} | '
                  f'fast-edge {sm["fast_edge_frac"]:.2f} | interior peaks {np.array2string(sm["peak_D"], precision=3)} '
                  f'(mass {np.round(sm["peak_mass"], 2)}), centroid {sm["centroid_D"]:.3g} | resid {resid:.5f}')
        n_pk = [len(r['s']['peak_D']) for r in runs]
        stable = len(set(n_pk)) == 1 and all(
            np.allclose(np.log10(r['s']['peak_D']), np.log10(runs[0]['s']['peak_D']), atol=0.15)
            for r in runs) if n_pk[0] else False
        best = min(runs, key=lambda r: r['resid'])
        fits = best['resid'] <= 2 * noise
        if not fits:
            verdict = (f'NO model in this family fits to the noise (best resid {best["resid"]:.4f} vs '
                       f'2x noise {2*noise:.4f}): systematics dominate -> trust only the interior '
                       f'centroid ({best["s"]["centroid_D"]:.3g} um^2/s) as an order of magnitude')
        elif n_pk[0] > 1 and stable and best['halves_same']:
            verdict = f'{n_pk[0]} interior populations, stable across lambda and both halves'
        elif n_pk[0] >= 1:
            verdict = ('ONE broad interior distribution (peak splitting is regulariser-dependent)'
                       if not stable else f'{n_pk[0]} interior peak(s), stable')
        else:
            verdict = 'no interior mass: dynamics outside the resolvable D window'
        print(f'  VERDICT: {verdict}')
        out[tag] = dict(n_px=n_px, noise=noise, D_single=D1, resid_single=r1, g=g.tolist(),
                        runs=[dict(lam=r['lam'], white=r['nug'], slow_edge=r['s']['slow_edge_frac'],
                                   fast_edge=r['s']['fast_edge_frac'], peaks=r['s']['peak_D'].tolist(),
                                   peak_mass=r['s']['peak_mass'], centroid=r['s']['centroid_D'],
                                   resid=r['resid']) for r in runs],
                        p_best=best['p'].tolist(), lam_best=best['lam'],
                        verdict=verdict)
        for r, ls in zip(runs, ('-', '--', ':')):
            axp.semilogx(D_um, r['p'] / max(r['p'].max(), 1e-30), ls, label=f'lam {r["lam"]:.0e}')
        axp.axvspan(D_um[0], D_um[2], color='0.85'); axp.axvspan(D_um[-3], D_um[-1], color='0.85')
        axp.axvline(0.92, color='gray', lw=0.8, ls=':')
        axp.set_title(f'{tag} ({n_px} px)', fontsize=9); axp.set_xlabel('D [um^2/s]')
    axs[0].set_ylabel('P(D) (norm.)'); axs[0].legend(fontsize=7)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.show()
    return out


@guard('D6')
def _d6():
    _d6_kernel()

    # ── 1) GEVD noise filter + clustering ──
    # Features (D6_FEATURES), all cross-half:
    #   'G+CV2' : full G(tau) curve + log CV2           (deliverable Section 3; default)
    #   'XY'    : tau map X = log10(1/gamma) + log CV2   (deliverable Section 2 reliability-max:
    #             two channels together cancel their shared noise)
    #   'G'     : G(tau) only                            (previous D6)
    # All three are scored and printed; D6_FEATURES picks the one used for the groups.
    G, G1, G2 = CACHE['G'], CACHE['G1'], CACHE['G2']
    _lg = lambda a: np.log10(np.clip(a, 1e-12, None))
    Y, Y1, Y2 = _lg(CACHE['dens']), _lg(CACHE.get('g1d', CACHE['dens'])), _lg(CACHE.get('g2d', CACHE['dens']))
    X, X1, X2 = (_lg(1.0 / np.clip(CACHE[k], 1e-6, None)) for k in ('gamma', 'g1g', 'g2g'))
    nuc = (CACHE['nuc'] & np.isfinite(G).all(-1) & np.isfinite(G1).all(-1) & np.isfinite(G2).all(-1)
           & np.isfinite(Y) & np.isfinite(Y1) & np.isfinite(Y2) & np.isfinite(X) & np.isfinite(X1) & np.isfinite(X2))
    if 'g1d' not in CACHE:
        print('note: half-record CV2 maps not cached -> CV2 halves = full record (CV2 looks more reproducible than it is)')
    feats = {
        'G':     (G1[nuc], G2[nuc], G[nuc]),
        'G+CV2': tuple(np.concatenate([g[nuc], y[nuc][:, None]], 1) for g, y in ((G1, Y1), (G2, Y2), (G, Y))),
        'XY':    tuple(np.stack([x[nuc], y[nuc]], 1) for x, y in ((X1, Y1), (X2, Y2), (X, Y))),
    }
    def _gevd_filter(F1, F2, Ff):
        F1, F2, Ff = (np.asarray(a, float) for a in (F1, F2, Ff))
        mu = (F1.mean(0) + F2.mean(0)) / 2; sd = (F1.std(0) + F2.std(0)) / 2 + 1e-9
        a1, a2, af = (F1 - mu) / sd, (F2 - mu) / sd, (Ff - mu) / sd
        Cg = (a1.T @ a2) / len(a1); Cg = (Cg + Cg.T) / 2
        Ct = np.cov(np.vstack([a1, a2]).T)
        ev, U = np.linalg.eigh(Ct); Wh = U / np.sqrt(np.clip(ev, 1e-12, None))     # whiten Ct
        lam_g, V = np.linalg.eigh(Wh.T @ Cg @ Wh); o = np.argsort(lam_g)[::-1]
        lam_g, Wg = lam_g[o], (Wh @ V)[:, o]
        keep = np.where(lam_g >= D6_REL_MIN)[0]
        if len(keep) == 0: keep = np.array([0])
        P1, P2, Pf = a1 @ Wg[:, keep], a2 @ Wg[:, keep], af @ Wg[:, keep]
        dis = ((P1 - P2) ** 2).sum(1) / (((P1 ** 2).sum(1) + (P2 ** 2).sum(1)) + 1e-12)
        return lam_g, keep, Pf, dis
    filt = {k: _gevd_filter(*v) for k, v in feats.items()}
    goods = {k: f[3] <= np.quantile(f[3], 1 - D6_DROP_FRAC) for k, f in filt.items()}
    for k, f in filt.items():
        print(f'GEVD [{k:5s}] reproducible fraction per axis: {np.round(f[0][:6], 2)} -> keep {len(f[1])}')
    ks = list(goods)
    print('kept-pixel overlap (Jaccard): ' + ', '.join(
        f'{a}/{b} {np.mean(goods[a] & goods[b]) / max(np.mean(goods[a] | goods[b]), 1e-12):.2f}'
        for i, a in enumerate(ks) for b in ks[i + 1:]))
    lam_g, keep, Pf, disagree = filt[D6_FEATURES]
    print(f'using D6_FEATURES = {D6_FEATURES!r}')
    if D6_FEATURES == 'XY' and Pf.shape[1] < 2:
        Pf = np.concatenate([Pf, filt['G'][2][:, :1]], 1)   # 1 kept XY axis: add G axis 1 for clustering
    good = disagree <= np.quantile(disagree, 1 - D6_DROP_FRAC)
    print(f'pixels: {len(good)} in nucleus -> {int(good.sum())} kept after dropping the '
          f'{D6_DROP_FRAC:.0%} least reproducible')
    # k-means (numpy) on the kept axes, kept pixels only
    rng = np.random.default_rng(0); Z = Pf[good]
    C = Z[rng.choice(len(Z), D6_N_CLUSTERS, replace=False)]
    for _ in range(100):
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C_new = np.array([Z[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(D6_N_CLUSTERS)])
        if np.allclose(C_new, C): break
        C = C_new
    order = np.argsort(C[:, 0]); lab = np.argsort(order)[lab]       # sort groups along axis 1
    Hh, Ww = nuc.shape
    lab_map = np.full((Hh, Ww), np.nan); good_map = np.zeros((Hh, Ww), bool)
    idx = np.flatnonzero(nuc); good_map.flat[idx[good]] = True; lab_map.flat[idx[good]] = lab
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].bar(np.arange(len(lam_g)) + 1, lam_g); ax[0].axhline(D6_REL_MIN, color='r', ls='--', lw=0.8)
    ax[0].set_xlabel('GEVD axis'); ax[0].set_ylabel('reproducible fraction'); ax[0].set_title('noise axes below the line')
    dm = np.full((Hh, Ww), np.nan); dm.flat[idx] = disagree
    im = ax[1].imshow(dm, cmap='magma'); plt.colorbar(im, ax=ax[1]); ax[1].set_title('half1 vs half2 disagreement')
    im = ax[2].imshow(lab_map, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    ax[2].set_title(f'{D6_N_CLUSTERS} groups (kept pixels)')
    for a in ax[1:]: a.axis('off')
    plt.tight_layout(); plt.show()

    # ── 2) P(D) per group ──
    groups = [('filtered nucleus', good_map)] + [(f'group {k+1}', lab_map == k) for k in range(D6_N_CLUSTERS)]
    out = {'gevd_rel': lam_g.tolist(), 'n_axes_kept': int(len(keep)), 'n_px_kept': int(good.sum())}
    out.update(_pd_groups(groups, 'P(D), iPSF kernel. grey = grid edges (not populations); dotted = paper D* 0.92'))
    # same filtered-nucleus P(D) under the other two filters, to see if the choice matters
    Hh_, Ww_ = nuc.shape; idx_ = np.flatnonzero(nuc); alt = []
    for k in ks:
        if k == D6_FEATURES: continue
        gm = np.zeros((Hh_, Ww_), bool); gm.flat[idx_[goods[k]]] = True
        alt.append((f'filtered nucleus [{k}]', gm))
    print('\n--- filter comparison: same CONTIN on the pixels kept by the other feature sets ---')
    out['filter_compare'] = _pd_groups(alt, 'P(D) of the filtered nucleus under the other noise filters')
    out['features'] = D6_FEATURES
    CACHE['d6_good_map'] = good_map
    out['verdict'] = 'ok'
    RESULT['D6'] = out""")

# ─────────────────────────────── cell 9c : D7 ──────────────────────────────
code(r"""# ══ D7 — ICA-defined ROIs (puncta / large structures / background) -> P(D) + alpha/gamma ══
# Rebuilds the deliverable's Section 3 ICA (cross-half GEVD on [G(tau), log CV2] -> de-ramp
# axis 2 -> FastICA, sparse 'condensate' component first, nucleoli forced negative), then
# cuts ROIs out of ONE component:
#   puncta            : connected blobs above the threshold with area < PUNCTA_MAX_PX
#   large structure k : the N_LARGE biggest blobs above the threshold (e.g. the two half-ellipses)
#   background        : the rest of the nucleus, minus nucleoli and a 2-px guard band around ROIs
# Check the overlay figure FIRST: if the half-ellipses are on the other component or the other
# sign, change ICA_ROI_COMP / ICA_ROI_SIGN and re-run only this cell (no reload needed).
# Or draw your own: ICA_ROI_NPY = path to an int label map (H, W): 0 = none, 1..n = ROIs.
ICA_ROI_COMP  = 'condensate'   # 'condensate' (sparse comp) or 'other'
ICA_ROI_SIGN  = +1             # +1: positive tail, -1: negative tail
ICA_ROI_Z     = 2.0            # threshold in nucleus z-score
PUNCTA_MAX_PX = 60             # blobs smaller than this are 'puncta' (binned px)
N_LARGE       = 2              # largest blobs kept as separate ROIs
ICA_ROI_NPY   = None

from scipy.ndimage import label as _cc_label, binary_dilation
from scipy.stats import kurtosis as _kurt

@guard('D7')
def _d7():
    from sklearn.decomposition import FastICA
    nuc = CACHE['nuc']
    G, G1, G2 = CACHE['G'], CACHE['G1'], CACHE['G2']
    _lg = lambda a: np.log10(np.clip(a, 1e-12, None))
    if 'g1d' not in CACHE:      # session from an older notebook: no half-record CV2 maps
        print('note: half-record CV2 maps not cached -> using the full-record CV2 for both halves')
    Y = _lg(CACHE['dens'])
    Y1, Y2 = _lg(CACHE.get('g1d', CACHE['dens'])), _lg(CACHE.get('g2d', CACHE['dens']))
    m = nuc & np.isfinite(G).all(-1) & np.isfinite(G1).all(-1) & np.isfinite(G2).all(-1) \
        & np.isfinite(Y) & np.isfinite(Y1) & np.isfinite(Y2)
    feat = lambda g, yl: np.concatenate([g.astype(np.float64), yl[..., None].astype(np.float64)], -1)
    F1, F2, Ff = feat(G1, Y1)[m], feat(G2, Y2)[m], feat(G, Y)[m]
    mu = (F1.mean(0) + F2.mean(0)) / 2; sd = (F1.std(0) + F2.std(0)) / 2 + 1e-9
    a1, a2, af = (F1 - mu) / sd, (F2 - mu) / sd, (Ff - mu) / sd
    Cg = (a1.T @ a2) / len(a1); Cg = (Cg + Cg.T) / 2
    Ct = np.cov(np.vstack([a1, a2]).T)
    ev, U = np.linalg.eigh(Ct); Wh = U / np.sqrt(np.clip(ev, 1e-12, None))
    lg_, V = np.linalg.eigh(Wh.T @ Cg @ Wh); o = np.argsort(lg_)[::-1]
    pj = af @ (Wh @ V)[:, o][:, :2]
    nucleoli_v = Y[m] < np.percentile(Y[m], 12)
    for k in range(2):
        if pj[nucleoli_v, k].mean() > pj[~nucleoli_v, k].mean(): pj[:, k] = -pj[:, k]
    yy, xx = np.mgrid[0:nuc.shape[0], 0:nuc.shape[1]]
    xa = (xx[m] - xx[m].mean()) / xx[m].std(); ya = (yy[m] - yy[m].mean()) / yy[m].std()
    Dr = np.vstack([xa, ya, np.ones_like(xa)]).T
    ax2 = pj[:, 1] - Dr @ np.linalg.lstsq(Dr, pj[:, 1], rcond=None)[0]
    Sx = FastICA(2, random_state=0, whiten='unit-variance', max_iter=1000).fit_transform(
        np.stack([pj[:, 0], ax2], 1))
    Sx = Sx[:, np.argsort([-abs(_kurt(Sx[:, 0])), -abs(_kurt(Sx[:, 1]))])]
    comps = {}
    for name, v in (('condensate', Sx[:, 0].copy()), ('other', Sx[:, 1].copy())):
        if v[nucleoli_v].mean() > v[~nucleoli_v].mean(): v = -v
        z = (v - v.mean()) / (v.std() + 1e-12)
        full = np.full(nuc.shape, np.nan); full[m] = z; comps[name] = full
    CACHE['ica_comps'] = comps

    nucleoli = np.zeros_like(nuc); nucleoli[m] = nucleoli_v
    if ICA_ROI_NPY:
        lab = np.load(ICA_ROI_NPY).astype(int)
        rois = [(f'ROI {k}', (lab == k) & m) for k in range(1, lab.max() + 1)]
        src = f'user label map {ICA_ROI_NPY}'
    else:
        zc = comps[ICA_ROI_COMP] * ICA_ROI_SIGN
        hot = m & (np.nan_to_num(zc, nan=-np.inf) > ICA_ROI_Z)
        cc, n_cc = _cc_label(hot)
        sizes = np.bincount(cc.ravel())[1:] if n_cc else np.array([], int)
        order = np.argsort(sizes)[::-1]
        large = [i + 1 for i in order if sizes[i] >= PUNCTA_MAX_PX][:N_LARGE]
        small = [i + 1 for i in range(n_cc) if sizes[i] < PUNCTA_MAX_PX]
        rois = [('puncta', np.isin(cc, small))]
        rois += [(f'large structure {k+1}', cc == i) for k, i in enumerate(large)]
        rest_large = [i + 1 for i in order if sizes[i] >= PUNCTA_MAX_PX and (i + 1) not in large]
        if rest_large: rois.append(('other large blobs', np.isin(cc, rest_large)))
        src = (f'ICA {ICA_ROI_COMP}, sign {ICA_ROI_SIGN:+d}, z > {ICA_ROI_Z}: {n_cc} blobs '
               f'({len(small)} puncta < {PUNCTA_MAX_PX} px, sizes of largest: {sorted(sizes, reverse=True)[:5]})')
    any_roi = np.zeros_like(nuc)
    for _, r in rois: any_roi |= r
    bg = m & ~nucleoli & ~binary_dilation(any_roi, iterations=2)
    rois += [('nucleoli', nucleoli & ~any_roi), ('background', bg)]
    CACHE['d7_rois'] = rois
    print('ROIs from', src)

    # overlay: both components + the ROI label map
    lab_map = np.full(nuc.shape, np.nan)
    for k, (_, r) in enumerate(rois): lab_map[r] = k
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for a, name in zip(ax[:2], ('condensate', 'other')):
        im = a.imshow(comps[name], cmap='coolwarm', vmin=-3, vmax=3); plt.colorbar(im, ax=a, fraction=0.046)
        for k, (_, r) in enumerate(rois[:-2]):
            a.contour(r, levels=[0.5], colors=[f'C{k}'], linewidths=0.8)
        a.set_title(f'ICA {name} (z)' + ('  <- ROI source' if name == ICA_ROI_COMP and not ICA_ROI_NPY else ''),
                    fontsize=9); a.axis('off')
    im = ax[2].imshow(lab_map, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    ax[2].set_title('ROIs: ' + ', '.join(f'{k}={t}' for k, (t, _) in enumerate(rois)), fontsize=7)
    ax[2].axis('off'); plt.tight_layout(); plt.show()
    try: np.save(os.path.join(_out_dir(), 'd7_roi_labels.npy'), np.nan_to_num(lab_map, nan=-1).astype(int))
    except Exception as e: print('could not save ROI map:', e)

    # alpha / gamma per ROI: c0-weighted ROI curve, naive vs finite-T corrected (utils/finite_t.py)
    T = CACHE['T']; taus = np.asarray(RECON_TAUS)
    rows, curves = [], []
    for tag, r in rois:
        g, n_px = ik.roi_curve(G, CACHE['C0'], r)
        if n_px < 20: continue
        curves.append((tag, g)); rows.append((tag, n_px, r))
    out = {'source': src}
    if curves:
        fitc = ft.debias(np.array([c for _, c in curves]), taus, T, global_alpha=False)
        print(f'\n{"ROI":22s} {"px":>6s} {"gamma_pix(med)":>14s} {"alpha":>6s} {"gamma":>8s} '
              f'{"alpha_ft":>8s} {"gamma_ft":>8s} {"white(1-A)":>10s}')
        for i, (tag, n_px, r) in enumerate(rows):
            gp = float(np.nanmedian(CACHE['gamma'][r]))
            print(f'{tag:22s} {n_px:6d} {gp:14.4f} {fitc["alpha_raw"][i]:6.2f} {fitc["gamma_raw"][i]:8.4f} '
                  f'{fitc["alpha"][i]:8.2f} {fitc["gamma"][i]:8.4f} {1-fitc["A"][i]:10.2f}')
            out[tag] = dict(n_px=n_px, gamma_pix_median=gp,
                            alpha_naive=float(fitc['alpha_raw'][i]), gamma_naive=float(fitc['gamma_raw'][i]),
                            alpha_ft=float(fitc['alpha'][i]), gamma_ft=float(fitc['gamma'][i]),
                            white=float(1 - fitc['A'][i]))
        print('  alpha/gamma are per-ROI fits of A/(1+gamma tau^alpha) [gamma in 1/frame^alpha];'
              ' *_ft = finite-T corrected. Compare ROIs on the *_ft columns.')
        plt.figure(figsize=(6, 3.8))
        for tag, g in curves: plt.semilogx(taus, g, 'o-', ms=3, label=tag)
        plt.xlabel('tau [frames]'); plt.ylabel('G (c0-weighted ROI mean)'); plt.legend(fontsize=7)
        plt.title('ROI curves', fontsize=10); plt.tight_layout(); plt.show()

    # P(D) per ROI (same CONTIN as D6, same verdict rules)
    pd_out = _pd_groups([(t, r) for t, r in rois],
                        'D7: P(D) per ICA ROI. grey = grid edges; dotted = paper D* 0.92')
    for tag, v in pd_out.items():
        out.setdefault(tag, {}).update(v)
    out['verdict'] = 'ok'
    RESULT['D7'] = out""")

# ─────────────────────────────── cell 9d : D8 ──────────────────────────────
code(r"""# ══ D8 — fast / slow populations in space: per-pixel 2-population map ══
# Each pixel's G(tau>=1) is fitted as  a_f K(tau; D_fast) + a_s K(tau; D_slow)  with the SAME
# finite-T iPSF kernel columns as D6 (white noise only touches tau=0, so it drops out).
# f_fast = a_f / (a_f + a_s) is the share of the fluctuating signal that is fast.
#   * f map bimodal + spatially structured  -> two populations sit in different places
#   * f unimodal around a middle value       -> both populations coexist inside every pixel
# D_fast / D_slow default to the two largest interior peaks of the D6 'filtered nucleus'
# spectrum; set them by hand to override. Per-pixel noise is large, so G is lightly smoothed
# in space first (TWOPOP_SMOOTH_PX, 0 = off) -- the map is a picture, not a per-pixel claim.
D_FAST_UM = None          # um^2/s, None = from D6
D_SLOW_UM = None
TWOPOP_SMOOTH_PX = 1.0

@guard('D8')
def _d8():
    K_full, K_half, D_um = _d6_kernel()
    d6 = (RESULT.get('D6') or {}).get('filtered nucleus') or {}
    Df, Ds = D_FAST_UM, D_SLOW_UM
    if (Df is None or Ds is None) and d6:
        run = min(d6['runs'], key=lambda r: r['resid'])
        pk = sorted(zip(run['peak_mass'], run['peaks']), reverse=True)[:2]
        if len(pk) == 2:
            Ds_, Df_ = sorted(p for _, p in pk)
            Df = Df if Df is not None else Df_; Ds = Ds if Ds is not None else Ds_
    assert Df is not None and Ds is not None, 'set D_FAST_UM / D_SLOW_UM (D6 found < 2 peaks)'
    jf, js = int(np.argmin(abs(np.log(D_um / Df)))), int(np.argmin(abs(np.log(D_um / Ds))))
    kf, ks = K_full[:, jf], K_full[:, js]
    print(f'D_fast = {D_um[jf]:.3g} um^2/s, D_slow = {D_um[js]:.3g} um^2/s  '
          f'(ratio {D_um[jf]/D_um[js]:.0f}x; absolute values scale with the iPSF q, the ratio does not)')

    nuc = CACHE['nuc'] & np.isfinite(CACHE['G']).all(-1)
    G = CACHE['G'].astype(np.float64).copy()
    if TWOPOP_SMOOTH_PX:
        w = gaussian_filter(nuc.astype(float), TWOPOP_SMOOTH_PX)
        for k in range(G.shape[-1]):
            G[..., k] = gaussian_filter(np.where(nuc, G[..., k], 0.0), TWOPOP_SMOOTH_PX) / np.maximum(w, 1e-6)
    g = G[nuc]                                                    # (N, K)
    # 2x2 non-negative least squares, closed form with clipping
    M = np.array([[kf @ kf, kf @ ks], [kf @ ks, ks @ ks]]); b = np.stack([g @ kf, g @ ks], 1)
    a = np.linalg.solve(M, b.T).T
    neg_f, neg_s = a[:, 0] < 0, a[:, 1] < 0
    a[neg_f, 0] = 0; a[neg_f, 1] = np.clip(g[neg_f] @ ks / (ks @ ks), 0, None)
    a[neg_s, 1] = 0; a[neg_s, 0] = np.clip(g[neg_s] @ kf / (kf @ kf), 0, None)
    tot = a.sum(1)
    f = np.where(tot > 0, a[:, 0] / np.maximum(tot, 1e-12), np.nan)
    resid = np.sqrt(((g - a[:, :1] * kf - a[:, 1:] * ks) ** 2).mean(1))
    fmap = np.full(nuc.shape, np.nan); fmap[nuc] = f
    amap = np.full(nuc.shape, np.nan); amap[nuc] = tot
    CACHE['f_fast'] = fmap

    rois = CACHE.get('d7_rois') or []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    im = ax[0].imshow(fmap, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax[0], fraction=0.046, label='fast fraction f')
    for k, (t, r) in enumerate(rois[:-2]):
        ax[0].contour(r, levels=[0.5], colors='k', linewidths=0.6)
    ax[0].set_title(f'fast share  (red = D~{D_um[jf]:.2g}, blue = D~{D_um[js]:.2g} um^2/s)', fontsize=9)
    ax[0].axis('off')
    ax[1].hist(f[np.isfinite(f)], bins=50, color='0.4')
    for k, (t, r) in enumerate(rois):
        v = fmap[r & nuc]; v = v[np.isfinite(v)]
        if v.size: ax[1].axvline(np.median(v), color=f'C{k}', lw=1.5, label=f'{t}: {np.median(v):.2f}')
    ax[1].set_xlabel('fast fraction f'); ax[1].set_ylabel('pixels'); ax[1].legend(fontsize=7)
    ax[1].set_title('distribution (lines = ROI medians)', fontsize=9)
    d7 = RESULT.get('D7') or {}
    for k, (t, r) in enumerate(rois):
        if t in d7 and 'p_best' in d7[t]:
            pb = np.asarray(d7[t]['p_best'])
            ax[2].semilogx(D_um, pb / max(pb.max(), 1e-30), color=f'C{k}', label=t)
    if not any(t in d7 and 'p_best' in d7[t] for t, _ in rois) and 'p_best' in d6:
        pb = np.asarray(d6['p_best']); ax[2].semilogx(D_um, pb / pb.max(), 'k', label='filtered nucleus')
    for Dv in (D_um[jf], D_um[js]): ax[2].axvline(Dv, color='k', ls='--', lw=0.8)
    ax[2].axvspan(D_um[0], D_um[2], color='0.85'); ax[2].axvspan(D_um[-3], D_um[-1], color='0.85')
    ax[2].set_xlabel('D [um^2/s]'); ax[2].set_ylabel('P(D) (norm.)'); ax[2].legend(fontsize=7)
    ax[2].set_title('P(D) per ROI (D7); dashed = D_fast / D_slow', fontsize=9)
    plt.tight_layout(); plt.show()

    fv = f[np.isfinite(f)]
    hist, edges = np.histogram(fv, bins=20, range=(0, 1))
    sm = np.convolve(np.r_[hist[:2][::-1], hist, hist[-2:][::-1]], np.ones(5) / 5, 'valid')
    pk = [i for i in range(len(sm)) if sm[i] >= 0.3 * sm.max()
          and sm[i] >= sm[max(i - 1, 0)] and sm[i] >= sm[min(i + 1, len(sm) - 1)]]
    # two peaks only count as separate modes if the dip between them is < 70% of the lower one
    modes = pk[:1]
    for i in pk[1:]:
        if sm[modes[-1]:i + 1].min() < 0.7 * min(sm[modes[-1]], sm[i]): modes.append(i)
        elif sm[i] > sm[modes[-1]]: modes[-1] = i
    n_modes = len(modes)
    print(f'fast fraction: median {np.median(fv):.2f}, IQR {np.percentile(fv,25):.2f}-{np.percentile(fv,75):.2f}, '
          f'{n_modes} mode(s); per-pixel 2-pop residual median {np.median(resid):.4f}')
    out = dict(D_fast=float(D_um[jf]), D_slow=float(D_um[js]), f_median=float(np.median(fv)),
               f_iqr=[float(np.percentile(fv, 25)), float(np.percentile(fv, 75))], n_modes=n_modes)
    for t, r in rois:
        v = fmap[r & nuc]; v = v[np.isfinite(v)]
        if v.size:
            out[t] = float(np.median(v)); print(f'  {t:22s} median f = {np.median(v):.2f}  ({v.size} px)')
    print('VERDICT D8:', 'fast and slow populations are SPATIALLY SEPARATED (bimodal f)' if n_modes >= 2
          else 'fast and slow populations COEXIST within pixels (unimodal f); compare ROI medians for where each dominates')
    try: np.save(os.path.join(_out_dir(), 'd8_fast_fraction.npy'), fmap)
    except Exception as e: print('could not save f map:', e)
    out['verdict'] = 'ok'
    RESULT['D8'] = out""")

# ─────────────────────────────── cell 9e : D9 ──────────────────────────────
code(r"""# ══ D9 — is the slow P(D) peak real dynamics or drift? (temporal high-pass test) ══
# Same ROIs, same CONTIN, but on the HIGH-PASSED movie (cell 4) with a kernel that applies the
# identical filter (ik.ipsf_kernel(..., highpass_W=HP_W)). Data and kernel are filtered the same
# way, so for STATIONARY dynamics P(D) must come back unchanged (the filter only costs SNR).
# Slow drift / bleaching / cell motion is non-stationary: the kernel cannot model it, so if the
# slow peak was drift it shrinks or vanishes here.
@guard('D9')
def _d9():
    assert 'G_hp' in CACHE, 'no high-passed G (HP_W = 0 or the high-pass failed in cell 4)'
    _d6_kernel()
    q2, S, D_grid, dt = CACHE['d6_q2S']
    Th = CACHE['T_hp']
    K_hf, _ = ik.ipsf_kernel(RECON_TAUS, dt, q2, S, D=D_grid, T=Th, highpass_W=HP_W)
    K_hh, _ = ik.ipsf_kernel(RECON_TAUS, dt, q2, S, D=D_grid, T=Th // 2, highpass_W=HP_W)
    D_um = D_grid * 1e12
    # split slow / fast at the geometric mean of the D6 peaks (fallback 1 um^2/s)
    d6 = (RESULT.get('D6') or {}).get('filtered nucleus') or {}
    split = 1.0
    if d6:
        run = min(d6['runs'], key=lambda r: r['resid'])
        pk = sorted(p for _, p in sorted(zip(run['peak_mass'], run['peaks']), reverse=True)[:2])
        if len(pk) == 2: split = float(np.sqrt(pk[0] * pk[1]))
    rois = []
    if 'd6_good_map' in CACHE: rois.append(('filtered nucleus', CACHE['d6_good_map']))
    for t, r in (CACHE.get('d7_rois') or []):
        if t in ('nucleoli', 'background'): rois.append((t, r))
    if not rois: rois = [('nucleus', CACHE['nuc'])]
    print(f'slow/fast split at D = {split:.3g} um^2/s;  high-pass W = {HP_W} frames, T_eff = {Th}')
    orig = _pd_groups(rois, 'D9 reference: original movie')
    hp = _pd_groups(rois, f'D9: high-passed movie (W={HP_W}) with the matching kernel',
                    keys=('G_hp', 'G1_hp', 'G2_hp', 'C0_hp', 'C0_1_hp', 'C0_2_hp'),
                    kern=(K_hf, K_hh, D_um))

    def _slow(o):
        p = np.asarray(o['p_best']); p = p / max(p.sum(), 1e-30)
        sl = p[D_um < split]; m_s = float(sl.sum())
        pos = float(np.exp((np.log(D_um[D_um < split]) * sl).sum() / max(m_s, 1e-30))) if m_s > 0 else float('nan')
        fa = p[D_um >= split]
        pos_f = float(np.exp((np.log(D_um[D_um >= split]) * fa).sum() / max(fa.sum(), 1e-30))) if fa.sum() > 0 else float('nan')
        return m_s, pos, pos_f
    out = {'split': split, 'HP_W': HP_W}
    print(f'\n{"ROI":18s} {"slow mass":>18s} {"slow centroid":>22s} {"fast centroid":>22s}   (original -> high-passed)')
    verdicts = []
    for t, _ in rois:
        if t not in orig or t not in hp: continue
        mo, po, fo = _slow(orig[t]); mh, ph, fh = _slow(hp[t])
        print(f'{t:18s} {mo:8.2f} -> {mh:5.2f}   {po:9.3g} -> {ph:8.3g}   {fo:9.3g} -> {fh:8.3g}')
        # calibrated on simulation (two populations + random-walk drift): with NO drift the
        # high-pass alone costs ~30% of the slow mass at the same position; drift adds slow mass
        # and drags the original centroid to lower D, while the high-passed result stays put.
        keep = (mh >= 0.5 * mo) and np.isfinite(ph) and abs(np.log10(ph / po)) < 0.2
        if keep and np.log10(ph / po) > 0.1:
            print(f'{"":18s} (original slow centroid is pulled low by drift; use the high-passed value {ph:.3g})')
        verdicts.append(keep)
        out[t] = dict(slow_mass_orig=mo, slow_mass_hp=mh, slow_pos_orig=po, slow_pos_hp=ph,
                      fast_pos_orig=fo, fast_pos_hp=fh, survives=bool(keep))
    if verdicts and all(verdicts):
        v = ('SLOW PEAK IS REAL — it survives the high-pass at the same position, so it is stationary '
             'dynamics, not drift. Quote the high-passed slow centroid (drift-free).')
    elif verdicts and not any(verdicts):
        v = ('SLOW PEAK IS (MOSTLY) DRIFT — it loses mass or moves once slow drift is filtered out; '
             'treat the slow population as an artefact of non-stationarity.')
    else:
        v = 'MIXED — the slow peak survives in some ROIs and not others; see the table.'
    print('\nVERDICT D9:', v)
    out['verdict'] = 'ok'; out['summary'] = v
    RESULT['D9'] = out""")

# ─────────────────────────────── cell 9f : D10 ─────────────────────────────
code(r"""# ══ D10 — tile CONTIN map: where are the slow / fast peaks, and how much fast mass? ══
# Per-pixel CONTIN is impossible (~4% noise per pixel; two peaks need ~0.1-0.3%). Tiles of
# TILE_PX x TILE_PX pixels (c0-weighted, ratio of means) bring the noise to ~0.3-0.5%, enough for
# a coarse map of the two peak positions and the fast mass share. Each tile is also inverted on
# both halves; the half1-vs-half2 correlation of each map (printed) says how much of it is real.
@guard('D10')
def _d10():
    K_full, K_half, D_um = _d6_kernel()
    lam = 1e-3
    nuc = CACHE['nuc'] & np.isfinite(CACHE['G']).all(-1)
    H, W = nuc.shape; tp = TILE_PX
    d6 = (RESULT.get('D6') or {}).get('filtered nucleus') or {}
    split = (RESULT.get('D9') or {}).get('split', 1.0)
    if split == 1.0 and d6:
        run = min(d6['runs'], key=lambda r: r['resid'])
        pk = sorted(p for _, p in sorted(zip(run['peak_mass'], run['peaks']), reverse=True)[:2])
        if len(pk) == 2: split = float(np.sqrt(pk[0] * pk[1]))
    lo, hi = D_um < split, D_um >= split
    ny, nx = H // tp, W // tp
    maps = {k: np.full((ny, nx), np.nan) for k in ('slow_D', 'fast_D', 'fast_share', 'fs_h1', 'fs_h2', 'white')}

    def _feat(p):
        p = p / max(p.sum(), 1e-30)
        ms, mf = p[lo].sum(), p[hi].sum()
        sD = np.exp((np.log(D_um[lo]) * p[lo]).sum() / ms) if ms > 1e-3 else np.nan
        fD = np.exp((np.log(D_um[hi]) * p[hi]).sum() / mf) if mf > 1e-3 else np.nan
        return sD, fD, mf / max(ms + mf, 1e-30)
    t0_ = time.time(); n_t = 0
    for iy in range(ny):
        for ix in range(nx):
            m = np.zeros_like(nuc); m[iy*tp:(iy+1)*tp, ix*tp:(ix+1)*tp] = True; m &= nuc
            if m.sum() < 0.5 * tp * tp: continue
            g, _ = ik.roi_curve(CACHE['G'], CACHE['C0'], m)
            g1, _ = ik.roi_curve(CACHE['G1'], CACHE['C0_1'], m)
            g2, _ = ik.roi_curve(CACHE['G2'], CACHE['C0_2'], m)
            p, nug = ik.invert_kernel(g, K_full, lam=lam, return_nugget=True, n_iter=5000)
            sD, fD, fs = _feat(p)
            maps['slow_D'][iy, ix], maps['fast_D'][iy, ix], maps['fast_share'][iy, ix] = sD, fD, fs
            maps['white'][iy, ix] = nug
            maps['fs_h1'][iy, ix] = _feat(ik.invert_kernel(g1, K_half, lam=lam, n_iter=5000))[2]
            maps['fs_h2'][iy, ix] = _feat(ik.invert_kernel(g2, K_half, lam=lam, n_iter=5000))[2]
            n_t += 1
    print(f'{n_t} tiles of {tp}x{tp} px inverted ({time.time()-t0_:.0f}s); slow/fast split at {split:.3g} um^2/s')

    def _rel(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float('nan')
    r_fs = _rel(maps['fs_h1'], maps['fs_h2'])
    print(f'fast-share map, half1 vs half2 correlation r = {r_fs:.2f}  '
          f'(~0 = the map is noise; >0.5 = the spatial pattern is reproducible)')

    ext = (0, nx * tp, ny * tp, 0)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
    for a, k, cm, lab in ((ax[0], 'slow_D', 'viridis', 'slow peak D [um^2/s] (log)'),
                          (ax[1], 'fast_D', 'viridis', 'fast peak D [um^2/s] (log)'),
                          (ax[2], 'fast_share', 'coolwarm', 'fast mass share'),
                          (ax[3], 'white', 'magma', 'white-noise weight')):
        d = np.log10(maps[k]) if k.endswith('_D') else maps[k]
        im = a.imshow(d, cmap=cm, extent=ext, interpolation='nearest')
        a.contour(np.arange(W) + 0.5, np.arange(H) + 0.5, nuc, levels=[0.5], colors='k', linewidths=0.5)
        plt.colorbar(im, ax=a, fraction=0.046); a.set_title(lab, fontsize=9); a.axis('off')
    plt.suptitle(f'D10 tile CONTIN ({tp}x{tp} px, lam {lam:g}); fast-share half-split r = {r_fs:.2f}', fontsize=10)
    plt.tight_layout(); plt.show()
    for k in ('slow_D', 'fast_D', 'fast_share'):
        v = maps[k][np.isfinite(maps[k])]
        if v.size: print(f'  {k:10s}: median {np.median(v):.3g}, IQR {np.percentile(v,25):.3g}-{np.percentile(v,75):.3g}')
    try: np.savez(os.path.join(_out_dir(), 'd10_tile_maps.npz'), **maps, tile_px=tp, split=split)
    except Exception as e: print('could not save tile maps:', e)
    RESULT['D10'] = dict(verdict='ok', n_tiles=n_t, tile_px=tp, split=split, r_fast_share_halves=r_fs,
                         **{k + '_median': float(np.nanmedian(maps[k])) for k in ('slow_D', 'fast_D', 'fast_share')})""")

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
d4b = RESULT.get('D4b') or {}
if d4b.get('verdict') == 'ok':
    rows.append(('finite-T correction removes the D4 trend', d4b['fixed'],
                 f"bin span {d4b['span_naive']:.3f} -> {d4b['span_corr']:.3f} dec, "
                 f"d_alpha {d4b['d_alpha_naive']:+.3f} -> {d4b['d_alpha_corr']:+.3f}"))
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
if (RESULT.get('D9') or {}).get('summary'):
    print('  D9:', RESULT['D9']['summary'])
if (RESULT.get('D10') or {}).get('verdict') == 'ok':
    print(f"  D10: fast-share tile map half-split r = {RESULT['D10']['r_fast_share_halves']:.2f}")
for dk in ('D6', 'D7'):
    for tag, v in (RESULT.get(dk) or {}).items():
        if isinstance(v, dict) and 'verdict' in v:
            print(f'  {dk} {tag:18s}: {v["verdict"]}')

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
elif d4.get('trend') is not None and abs(d4.get('trend', 0)) >= 0.02 and \
        not (RESULT.get('D4b') or {}).get('fixed'):
    print('  * D4 says the finite-T bias is real and gamma-dependent, and D4b shows the')
    print('    exact finite-T correction does NOT remove all of it: look for slow drift /')
    print('    non-stationarity before trusting the slow end or the slow D6 peak.')
elif (RESULT.get('D4b') or {}).get('fixed'):
    print('  * D4b: the gamma-dependent bias is finite-T and is now corrected (D5 corrected')
    print('    rows; D6 has it in the kernel). Next: set IPSF_NPZ to the 3DGS COBRI iPSF')
    print('    (ipsf_cobri_xsec.ipynb) to put the D6 peaks on an absolute D scale.')
else:
    print('  * Nothing invalidating found. The measurement is what it was assumed to be,')
    print('    and the ~1-robust-axis conclusion stands. New axes need new acquisition')
    print('    (q-resolved DDM/STICS, or an orthogonal modality).')
save_results()
print('all outputs in', _out_dir())""")

# ─────────────────────────────── cell 11 : zip + download ──────────────────
code(r"""# ── zip every saved output (figures, log.txt, results.json) and download it ──
import shutil
from google.colab import files

# Get the active output directory
output_dir = _out_dir()
zip_output_path = '/content/iscors_diagnostic_outputs'

print(f'Compressing outputs from: {output_dir}')
shutil.make_archive(zip_output_path, 'zip', output_dir)

print('Zip file created successfully. Starting download...')
files.download(f'{zip_output_path}.zip')""")

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

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "iscors_one_shot_diagnostics.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "cells:", len(CELLS))