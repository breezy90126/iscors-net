"""Gradio front-end for the iSCORS classical (no-ML) condensation-axis pipeline.

Upload a video (.zip / .tif / .npz), pick one of the five axis groups from
iscors_deliverable.ipynb (Sections 1-3), get the same left/right image pair
that the notebook's comparison grid (cell 9) shows for that row.
"""
import os
import sys
import gc
import zipfile
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
import gradio as gr
from scipy.ndimage import gaussian_filter, zoom
from scipy.linalg import eigh as geigh
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.gpu_iscors_fit import gpu_fit_maps, compute_density, compute_g_norm_torch

MASK_PATH = 'data/condensation_mask.tif'
RECON_TAUS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
GAMMA_SCALE = 2.0
CONDENSATION_SLOPE = 3.0

# group label -> (left cmap, right cmap). Same colormap pair for every
# group; ICA comp 2's *values* (not the colormap) get negated below.
GROUP_CMAPS = {
    '1/D* + CV²':                    ('coolwarm', 'viridis'),
    'slope=3 condensation':          ('coolwarm', 'viridis'),
    'reliability-max axis（最大重複軸）': ('coolwarm', 'viridis'),
    'GEVD':                          ('coolwarm', 'viridis'),
    'ICA（de-nuisance）':             ('coolwarm', 'viridis'),
}
GROUP_NAMES = list(GROUP_CMAPS)


# ───────────────────────── loaders (same logic as the notebook) ─────────────
def load_video(path, fname=None, n_frames=2000, bin_factor=2, start_frame=0):
    path_l = str(path).lower()
    if path_l.endswith('.zip'):
        extract_dir = tempfile.mkdtemp(prefix='iscors_video_')
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)

        def _find(root, name):
            for dp, _, fs in os.walk(root):
                if name:
                    if name in fs:
                        return os.path.join(dp, name)
                else:
                    for fn in fs:
                        if fn.lower().endswith(('.tif', '.tiff')):
                            return os.path.join(dp, fn)
            return None
        vp = _find(extract_dir, fname)
        assert vp, f'zip 內找不到可用的 .tif（內容: {os.listdir(extract_dir)}）'
    else:
        vp = path

    if str(vp).lower().endswith('.npz'):
        with np.load(vp) as nz:
            key = fname if (fname and fname in nz.files) else nz.files[0]
            arr = nz[key]
        total, H0, W0 = arr.shape
        Hb, Wb = H0 // bin_factor, W0 // bin_factor
        start_frame = max(0, min(int(start_frame), total))
        n = max(0, min(int(n_frames), total - start_frame)); Hc, Wc = Hb * bin_factor, Wb * bin_factor
        raw = np.empty((n, Hb, Wb), np.float32)
        for s in range(0, n, 100):
            e = min(s + 100, n)
            ch = arr[start_frame + s:start_frame + e].astype(np.float32)
            raw[s:e] = ch[:, :Hc, :Wc].reshape(e - s, Hb, bin_factor, Wb, bin_factor).mean((2, 4))
        return raw

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
    return raw


def preprocess(raw):
    ff = raw / (np.median(raw, axis=0)[None] + 1e-10)
    for t in range(len(ff)): ff[t] /= (gaussian_filter(ff[t], 4) + 1e-10)
    return ff


def load_mask(mask_path, shape):
    mk = np.asarray(tifffile.imread(mask_path)).squeeze()
    if mk.ndim == 3: mk = mk[..., 0]
    if mk.shape != shape: mk = zoom(mk, (shape[0] / mk.shape[0], shape[1] / mk.shape[1]), order=0)
    bd = np.zeros(shape, bool); bd[0, :] = bd[-1, :] = bd[:, 0] = bd[:, -1] = True
    return mk == min(np.unique(mk), key=lambda z: (mk[bd] == z).mean())   # interior = nucleus


def _maps(seg, taus, gamma_scale):
    d, _ = compute_density(seg, min_cv=0.005)
    g, _c, _z = compute_g_norm_torch(seg, taus, norm='nor1', min_cv=0.005)
    f = gpu_fit_maps(seg, recon_taus=taus, n_components=1, global_alpha=True,
                     gamma_scale=gamma_scale, min_cv=0.005, n_steps=400, verbose=False)
    return f['gamma'].astype(np.float32), d.astype(np.float32), g.cpu().numpy().astype(np.float32)


def _gevd(a1, a2):
    Cg = (a1.T @ a2) / a1.shape[0]; Cg = (Cg + Cg.T) / 2
    v, w = geigh(Cg, np.cov(np.vstack([a1, a2]).T)); o = np.argsort(v)[::-1]
    return v[o], w[:, o]


# ───────────────────────── the 5 axis-group pipeline ─────────────────────────
def compute_axis_pairs(video_path, mask_path, n_frames, bin_factor, start_frame,
                       taus=RECON_TAUS, gamma_scale=GAMMA_SCALE, slope=CONDENSATION_SLOPE):
    raw = load_video(video_path, None, n_frames, bin_factor, start_frame)
    vp = preprocess(raw); del raw; gc.collect()
    h = vp.shape[0] // 2
    gamma, dens, G = _maps(vp, taus, gamma_scale)
    g1g, g1d, G1   = _maps(vp[:h], taus, gamma_scale)
    g2g, g2d, G2   = _maps(vp[h:], taus, gamma_scale)
    del vp; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass

    nuc = load_mask(mask_path, dens.shape) & np.isfinite(gamma)
    _lg = lambda a: np.log10(np.clip(a, 1e-12, None))
    X,  Y  = _lg(1 / np.clip(gamma, 1e-6, None)), _lg(dens)
    X1, Y1 = _lg(1 / np.clip(g1g, 1e-6, None)),   _lg(g1d)
    X2, Y2 = _lg(1 / np.clip(g2g, 1e-6, None)),   _lg(g2d)

    m  = nuc & np.isfinite(X)  & np.isfinite(Y)
    mm = nuc & np.isfinite(X1) & np.isfinite(Y1) & np.isfinite(X2) & np.isfinite(Y2)
    def full(vec_on_m):
        z = np.full(nuc.shape, np.nan); z[m] = vec_on_m; return z

    pairs = {}

    # 1/D* + CV2 — raw channels
    pairs['1/D* + CV²'] = [('1/D* (log)', full(X[m])), ('CV2 (log)', full(Y[m]))]

    # slope=3 condensation — paper replication
    b3 = float((Y[m] - slope * X[m]).mean())
    cond  = full(((X + slope * Y - slope * b3) / (1 + slope ** 2))[m])
    yint  = full((Y - slope * X)[m])
    pairs['slope=3 condensation'] = [('slope-3 condensation', cond),
                                     ('y-intercept density (Y-3X)', yint)]

    # reliability-max axis — cross-half GEVD in (X,Y)
    mX, mY = float(X[mm].mean()), float(Y[mm].mean())
    sX, sY = float(X[mm].std()), float(Y[mm].std())
    f1 = np.vstack([(X1[mm]-mX)/sX, (Y1[mm]-mY)/sY]).astype(np.float32)
    f2 = np.vstack([(X2[mm]-mX)/sX, (Y2[mm]-mY)/sY]).astype(np.float32)
    Csig = (f1 @ f2.T) / f1.shape[1]; Csig = (Csig + Csig.T) / 2
    Ctot = np.cov(np.hstack([f1, f2]))
    _, Gv = geigh(Csig, Ctot)
    v_rel  = Gv[:, -1]
    v_perp = np.array([-v_rel[1], v_rel[0]])
    def proj(vec, Xs, Ys): return vec[0]*(Xs-mX)/sX + vec[1]*(Ys-mY)/sY
    pairs['reliability-max axis（最大重複軸）'] = [
        ('reliability-max axis', full(proj(v_rel, X, Y)[m])),
        ('relmax-orthogonal axis', full(proj(v_perp, X, Y)[m])),
    ]
    del f1, f2, Csig, Ctot, Gv

    # GEVD + ICA(de-nuisance) — full-tau GEVD, de-ramp axis 2, re-run FastICA
    feat = lambda g, yl: np.concatenate([g.astype(np.float32), yl[..., None].astype(np.float32)], -1)
    F1, F2, Ff = feat(G1, Y1)[m], feat(G2, Y2)[m], feat(G, Y)[m]
    mu = (F1.mean(0) + F2.mean(0)) / 2; sd = (F1.std(0) + F2.std(0)) / 2 + 1e-9
    _, W = _gevd((F1 - mu) / sd, (F2 - mu) / sd)
    pj = ((Ff - mu) / sd) @ W

    Ym = Y[m]; nucleoli = Ym < np.percentile(Ym, 12)
    if pj[nucleoli, 0].mean() > pj[~nucleoli, 0].mean(): pj[:, 0] = -pj[:, 0]
    if pj[nucleoli, 1].mean() > pj[~nucleoli, 1].mean(): pj[:, 1] = -pj[:, 1]

    yy, xx = np.mgrid[0:nuc.shape[0], 0:nuc.shape[1]]
    xa = ((xx[m]-xx[m].mean())/xx[m].std()).astype(np.float32); ya = ((yy[m]-yy[m].mean())/yy[m].std()).astype(np.float32)
    Dr = np.vstack([xa, ya, np.ones_like(xa)]).T
    ramp = Dr @ np.linalg.lstsq(Dr, pj[:, 1], rcond=None)[0]
    axis2_clean = pj[:, 1] - ramp

    pj_clean = np.stack([pj[:, 0], axis2_clean], axis=1)
    S = FastICA(2, random_state=0, whiten='unit-variance', max_iter=1000).fit_transform(pj_clean)
    S = S[:, np.argsort([-abs(kurtosis(S[:, 0])), -abs(kurtosis(S[:, 1]))])]
    condensate, other = S[:, 0].copy(), S[:, 1].copy()
    if condensate[nucleoli].mean() > condensate[~nucleoli].mean(): condensate = -condensate
    if other[nucleoli].mean()      > other[~nucleoli].mean():      other      = -other

    pairs['GEVD'] = [('G(tau) GEVD axis 1', full(pj[:, 0])), ('G(tau) GEVD axis 2', full(pj[:, 1]))]
    # comp 2's sign is flipped for display (nucleoli-blue forced it negative
    # above; the notebook view of this axis reads better nucleoli-positive)
    pairs['ICA（de-nuisance）'] = [('ICA comp 1', full(other)), ('ICA comp 2 (condensate)', full(-condensate))]

    del F1, F2, Ff, W, S, Dr; gc.collect()
    return pairs, nuc


# ───────────────────────── rendering + gradio UI ─────────────────────────────
_STATE = {'pairs': {}, 'nuc': None}


def _zscore(d, nuc):
    s = d[nuc]; return (d - np.nanmean(s)) / (np.nanstd(s) + 1e-9)


def _plot(name, data, cmap):
    z = _zscore(data, _STATE['nuc'])
    fig, a = plt.subplots(figsize=(5, 5))
    im = a.imshow(z, cmap=cmap, vmin=-2, vmax=3)
    a.set_title(name, fontsize=11); a.axis('off')
    plt.colorbar(im, ax=a, fraction=0.046)
    return fig


def render_group(group_name):
    if not group_name or group_name not in _STATE['pairs']:
        return None, None
    (n1, d1), (n2, d2) = _STATE['pairs'][group_name]
    c1, c2 = GROUP_CMAPS.get(group_name, ('coolwarm', 'viridis'))
    return _plot(n1, d1, c1), _plot(n2, d2, c2)


def on_run(video_path, mask_path, n_frames, bin_factor, start_frame, group_name,
          progress=gr.Progress()):
    if not video_path:
        raise gr.Error('請上傳影片檔案 (.zip / .tif / .npz)')
    progress(0.05, desc='讀取影片並前處理...')
    pairs, nuc = compute_axis_pairs(video_path, mask_path or MASK_PATH,
                                    int(n_frames), int(bin_factor), int(start_frame))
    _STATE['pairs'], _STATE['nuc'] = pairs, nuc
    progress(0.9, desc='繪圖...')
    return render_group(group_name)


with gr.Blocks(title='iSCORS axes viewer') as demo:
    gr.Markdown(
        '# iSCORS 軸檢視器\n'
        '重現並延伸 Hsiao *et al.*, '
        '[*Probing chromatin condensation dynamics in live cells using '
        'interferometric scattering correlation spectroscopy*]'
        '(https://doi.org/10.1038/s42003-024-06457-2), '
        '**Communications Biology** 7:763 (2024)。\n\n'
        '**論文原本的呈現方式**（對應下面「1/D\\* + CV²」「slope=3 condensation」兩組）：'
        '把每個像素的動態光散射訊號變異量 V_DLS（本 app 稱 **CV²**）與表觀擴散係數倒數 '
        '**1/D\\***，畫成 log-log 散佈圖，資料點落在**斜率為 3** 的直線上'
        '（V_DLS ∝ (1/D\\*)³），並把資料點**投影到這條線上的位置**當作染色質凝聚程度指標——'
        '論文用它偵測轉錄抑制、ATP 耗竭等處理造成的凝聚態改變，並用奈米粒子膠體資料驗證同一個模型。\n\n'
        '**論文之外的延伸**（`reliability-max axis`、`GEVD`、`ICA（de-nuisance）`三組）：'
        '這個 repo 用 cross-half 可複現性檢驗（廣義特徵分解）額外找候選軸，'
        '結論是穩健、可複現的生物軸大約只有這一條（核仁/密度），細節見 `RETROSPECTIVE_session.md`。\n\n'
        '上傳一支 iSCAT 影片，挑一組軸看它的左右兩張圖。'
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.File(label='影片 (.zip / .tif / .npz)',
                               file_types=['.zip', '.tif', '.tiff', '.npz'], type='filepath')
            mask_in  = gr.File(label='[選用] 自訂 nucleus mask (.tif)，留空則用內建的',
                               file_types=['.tif', '.tiff'], type='filepath')
            with gr.Row():
                start_in   = gr.Number(value=0,    precision=0, label='起始 frame')
                nframes_in = gr.Number(value=5000, precision=0, label='幀數')
                bin_in     = gr.Number(value=2,    precision=0, label='bin factor')
            run_btn  = gr.Button('載入並計算', variant='primary')
        with gr.Column(scale=2):
            group_dd = gr.Dropdown(GROUP_NAMES, value=GROUP_NAMES[0], label='要看的軸')
            with gr.Row():
                plot1 = gr.Plot(label='左')
                plot2 = gr.Plot(label='右')

    run_btn.click(on_run, [video_in, mask_in, nframes_in, bin_in, start_in, group_dd], [plot1, plot2])
    group_dd.change(render_group, group_dd, [plot1, plot2])

if __name__ == '__main__':
    demo.launch()
