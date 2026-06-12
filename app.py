"""Gradio inference frontend for iSCORS-Net — γ/α inference, physical-unit
conversion (τ_D, D_α) and comparison against iSCORS MATLAB GT (.mat)."""
import os
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import pearsonr, spearmanr
import gradio as gr

from datasets.phys_recon_dataset import PhysReconDataset
from models.pissl_tau_encoder import PISSLTauEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GAMMA_KEYS = ['D_map', 'Dmap', 'D', 'gamma', 'Gamma', 'GAMMA',
              'gamma_map', 'GammaMap', 'diffusion', 'Diffusion']


# ───────────────────────────── helpers ──────────────────────────────────────
def _parse_taus(s):
    return tuple(int(t.strip()) for t in s.split(',') if t.strip())


def preprocess_video(video_path, bin_factor, n_frames_max, chunk_size=100, progress=None):
    """Replicates p2-preprocess: chunked binning → flat-field → per-frame BG removal."""
    frame0 = tifffile.imread(video_path, key=0).astype(np.float32)
    H_orig, W_orig = frame0.shape
    H_bin, W_bin = H_orig // bin_factor, W_orig // bin_factor

    with tifffile.TiffFile(video_path) as tf:
        total_frames = len(tf.pages)
    n_frames = min(int(n_frames_max), total_frames)

    video_raw = np.empty((n_frames, H_bin, W_bin), dtype=np.float32)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = tifffile.imread(video_path, key=range(start, end)).astype(np.float32)
        T_c = end - start
        video_raw[start:end] = (chunk
            .reshape(T_c, H_bin, bin_factor, W_bin, bin_factor)
            .mean(axis=(2, 4)))
        del chunk
        if progress is not None:
            progress(0.05 + 0.35 * end / n_frames, desc=f'讀取影格 {end}/{n_frames}')

    T, H, W = video_raw.shape

    median_xy = np.median(video_raw, axis=0)
    video_ff = video_raw / (median_xy[np.newaxis] + 1e-10)

    video_proc = np.empty_like(video_ff)
    for t in range(T):
        bg = gaussian_filter(video_ff[t], sigma=4)
        video_proc[t] = video_ff[t] / (bg + 1e-10)
        if progress is not None and t % max(1, T // 10) == 0:
            progress(0.40 + 0.20 * t / T, desc=f'背景去除 {t}/{T}')

    return video_proc


def run_inference(video_proc, ckpt_path, recon_taus, gamma_scale, use_sigma,
                  n_components=1, fix_alpha=False, global_alpha=False, progress=None):
    """Replicates p2-inference: build eval dataset → load checkpoint → full-frame forward pass.

    n_components=2 loads a two-component (shared-α) checkpoint; the 4-channel output
    [f, γ_slow, γ_fast, α] is reduced to the effective γ = f·γ_fast+(1-f)·γ_slow and α.
    """
    if progress is not None:
        progress(0.62, desc='建立資料集 (G_empirical, σ_G)...')
    infer_ds = PhysReconDataset(video_tensor=video_proc, recon_taus=recon_taus,
                                patch_size=64, mode='eval')
    cell_mask = infer_ds.cell_mask

    model = PISSLTauEncoder(recon_taus=recon_taus, predict_amplitude=False,
                            gamma_scale=gamma_scale, use_sigma=use_sigma,
                            n_components=n_components, fix_alpha=fix_alpha,
                            global_alpha=global_alpha).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    full_inp = infer_ds[0]                       # (K, H, W)
    K_inf, Hv, Wv = full_inp.shape
    ph = (8 - Hv % 8) % 8
    pw = (8 - Wv % 8) % 8
    if ph or pw:
        full_inp = F.pad(full_inp, (0, pw, 0, ph))

    full_sigma = torch.from_numpy(
        infer_ds.sigma_g_norm[:Hv, :Wv].transpose(2, 0, 1).astype(np.float32))
    if ph or pw:
        full_sigma = F.pad(full_sigma, (0, pw, 0, ph))

    if progress is not None:
        progress(0.75, desc='模型推論中...')
    with torch.no_grad():
        preds_out = model(full_inp.unsqueeze(0).to(DEVICE),
                          sigma_g_norm=full_sigma.unsqueeze(0).to(DEVICE))

    if n_components == 2:
        f_o, gs_o, gf_o, a_o = (preds_out[:, 0:1], preds_out[:, 1:2],
                                preds_out[:, 2:3], preds_out[:, 3:4])
        gamma_t = f_o * gf_o + (1.0 - f_o) * gs_o          # effective γ
        gamma_pred = gamma_t[0, 0].cpu().numpy()[:Hv, :Wv]
        alpha_pred = a_o[0, 0].cpu().numpy()[:Hv, :Wv]
    else:
        gamma_pred = preds_out[0, 0].cpu().numpy()[:Hv, :Wv]
        alpha_pred = preds_out[0, 1].cpu().numpy()[:Hv, :Wv]
    gamma_pred[~cell_mask] = np.nan
    alpha_pred[~cell_mask] = np.nan
    return infer_ds, cell_mask, gamma_pred, alpha_pred


def compute_physical_units(gamma_pred, alpha_pred, cell_mask, l_psf, pixel_size_nm, frame_rate_hz):
    """Replicates p2-physical-units: (γ,α) → τ_D = γ^(-1/α) → D_α via generalised Stokes-Einstein."""
    eps_pu = 1e-10
    tau_D_frames = np.power(np.clip(gamma_pred, eps_pu, None),
                            -1.0 / np.clip(alpha_pred, eps_pu, None))
    tau_D_frames[~cell_mask] = np.nan

    w_phys_nm = l_psf * pixel_size_nm

    if frame_rate_hz is not None:
        dt_s = 1.0 / frame_rate_hz
        time_map = tau_D_frames * dt_s
        D_alpha = (w_phys_nm**2) / (4.0 * np.power(time_map, alpha_pred) + eps_pu)
        time_lbl, d_unit = 's', 'nm² / s^α'
    else:
        time_map = tau_D_frames
        D_alpha = (l_psf**2) / (4.0 * np.power(time_map, alpha_pred) + eps_pu)
        time_lbl, d_unit = 'frame-lags', 'px² / frame^α'

    D_alpha[~cell_mask] = np.nan
    return time_map, D_alpha, time_lbl, d_unit, w_phys_nm


def _load_mat(mat_path):
    try:
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        mat_keys = [k for k in mat.keys() if not k.startswith('_')]
    except NotImplementedError:                                  # HDF5 / v7.3 .mat
        import h5py
        mat = {}
        with h5py.File(mat_path, 'r') as hf:
            for k in hf.keys():
                mat[k] = np.array(hf[k])
        mat_keys = list(mat.keys())
    return mat, mat_keys


def _align(arr, Hm, Wm):
    if arr is None:
        return None
    if (arr.shape[0], arr.shape[1]) != (Hm, Wm):
        return zoom(arr, (Hm / arr.shape[0], Wm / arr.shape[1]), order=1).astype(np.float32)
    return arr.astype(np.float32)


def _coord(arr, Hm, Wm):
    """MATLAB → Python axis convention: flip vertically then rotate 90° clockwise."""
    if arr is None:
        return None
    out = np.rot90(np.flipud(arr), k=-1).astype(np.float32)
    if out.shape != (Hm, Wm):
        out = zoom(out, (Hm / out.shape[0], Wm / out.shape[1]), order=1).astype(np.float32)
    return out


def compare_with_gt(mat_path, gamma_pred, cell_mask, gt_gamma_key=None):
    """Replicates p2-gt-compare γ branch. GT field is D (iSCORS convention).

    PHYSICS: γ is the decay rate of G(τ)=1/(1+γτ^α), so faster diffusion
    (larger D) → larger γ → γ ∝ D. We compare model γ vs D_map DIRECTLY
    (expect POSITIVE corr). The earlier 1/D inversion was a sign error that
    flipped Pearson negative and made a working model look broken.
    iSCORS GT has no true α field.
    """
    mat, mat_keys = _load_mat(mat_path)
    Hm, Wm = gamma_pred.shape

    gt_gamma_raw = None
    for k in ([gt_gamma_key] if gt_gamma_key else GAMMA_KEYS):
        if k in mat and hasattr(mat[k], 'shape') and mat[k].squeeze().ndim == 2:
            gt_gamma_raw = mat[k].astype(np.float64).squeeze()
            break
    if gt_gamma_raw is None:
        raise ValueError(f'.mat 中找不到可用的 γ/D 欄位 (可用 keys: {mat_keys})；'
                         f'已嘗試: {GAMMA_KEYS}')

    Hg, Wg = gt_gamma_raw.shape
    if abs(Hg - Wm) < abs(Hg - Hm) and Hg != Hm:
        gt_gamma_raw = gt_gamma_raw.T

    gt_gamma_r = _align(gt_gamma_raw, Hm, Wm)
    gt_gamma_r = _coord(gt_gamma_r, Hm, Wm)

    # γ ∝ D — compare against D_map directly (no 1/D inversion).
    gt_gamma_r[~cell_mask] = np.nan

    ok_g = (cell_mask & np.isfinite(gt_gamma_r) & np.isfinite(gamma_pred) & (gt_gamma_r > 0))
    gg = gt_gamma_r[ok_g].astype(float)
    mg = gamma_pred[ok_g].astype(float)
    pg, _ = pearsonr(gg, mg)
    sg = spearmanr(gg, mg).statistic
    eg = np.abs(mg - gg).mean()
    # scale-free MAE: γ (dimensionless rate) and D (physical units) live on
    # different scales, so raw MAE is dominated by the scale gap. z-score both.
    zg = (gg - gg.mean()) / (gg.std() + 1e-10)
    zm = (mg - mg.mean()) / (mg.std() + 1e-10)
    eg_z = float(np.abs(zm - zg).mean())

    stats = dict(pearson=float(pg), spearman=float(sg), mae=float(eg), mae_z=eg_z,
                 n=int(ok_g.sum()), gt_mean=float(gg.mean()), model_mean=float(mg.mean()))
    return gt_gamma_r, stats


# ───────────────────────────── plotting ─────────────────────────────────────
def _imshow_panel(ax, data, cmap, title, pct=(1, 99)):
    p1, p99 = np.nanpercentile(data, pct[0]), np.nanpercentile(data, pct[1])
    im = ax.imshow(data, cmap=cmap, vmin=p1, vmax=p99)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def make_maps_figure(gamma_pred, alpha_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _imshow_panel(axes[0], gamma_pred, 'magma', 'γ (model-native, dimensionless)')
    _imshow_panel(axes[1], alpha_pred, 'plasma', 'α (model-native, dimensionless)')
    fig.suptitle('Model Inference — γ / α maps  (1st–99th percentile, cell mask only)', fontsize=13)
    fig.tight_layout()
    return fig


def make_physical_figure(gamma_pred, time_map, D_alpha, time_lbl, d_unit, l_psf, w_phys_nm):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _imshow_panel(axes[0], gamma_pred, 'magma', 'γ (model-native, dimensionless)')
    _imshow_panel(axes[1], time_map, 'viridis', f'τ_D  [{time_lbl}]')
    _imshow_panel(axes[2], D_alpha, 'cividis', f'D_α  [{d_unit}]')
    fig.suptitle(f'Physical-unit conversion (post-hoc)  PSF w={l_psf:.2f}px = {w_phys_nm:.1f}nm',
                 fontsize=13)
    fig.tight_layout()
    return fig


def make_gt_figure(gt_gamma_r, gamma_pred, stats):
    diff = np.abs(gamma_pred - gt_gamma_r)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _imshow_panel(axes[0], gt_gamma_r, 'magma', 'γ iSCORS GT  (D_map)')
    _imshow_panel(axes[1], gamma_pred, 'magma', 'γ model')
    _imshow_panel(axes[2], diff, 'hot', f'|Δγ| (raw)  z-MAE={stats["mae_z"]:.3f}')
    fig.suptitle(f'Model vs iSCORS GT — γ vs D (expect +)   '
                 f'Pearson={stats["pearson"]:+.3f}  Spearman={stats["spearman"]:+.3f}', fontsize=13)
    fig.tight_layout()
    return fig


# ───────────────────────────── pipeline ─────────────────────────────────────
def run_pipeline(video_file, ckpt_file, mat_file,
                 recon_taus_str, bin_factor, n_frames, gamma_scale, use_sigma,
                 n_components, alpha_mode, wavelength_nm, na, pixel_size_nm, frame_rate_hz,
                 progress=gr.Progress()):
    if video_file is None or ckpt_file is None:
        raise gr.Error('請至少上傳影片 (.tif) 與模型權重 (.pth)')

    recon_taus = _parse_taus(recon_taus_str)
    bin_factor = int(bin_factor)
    n_frames   = int(n_frames)
    pixel_size_eff_nm = float(pixel_size_nm) * bin_factor   # raw camera px → binned px size
    l_psf = 0.61 * float(wavelength_nm) / float(na) / pixel_size_eff_nm

    work_dir = tempfile.mkdtemp(prefix='iscors_app_')

    progress(0.02, desc='讀取並前處理影片...')
    video_proc = preprocess_video(video_file, bin_factor, n_frames, progress=progress)

    progress(0.60, desc='建立資料集並執行模型推論...')
    infer_ds, cell_mask, gamma_pred, alpha_pred = run_inference(
        video_proc, ckpt_file, recon_taus, float(gamma_scale), bool(use_sigma),
        n_components=int(n_components),
        fix_alpha=(alpha_mode == 'fixed α=1'),
        global_alpha=(alpha_mode == 'global α (shared scalar)'),
        progress=progress)

    gc, ac = gamma_pred[cell_mask], alpha_pred[cell_mask]
    lines = [
        f'Cell pixels: {int(cell_mask.sum())} / {cell_mask.size} ({100*cell_mask.mean():.1f}%)',
        f'γ  mean={gc.mean():.4f}  std={gc.std():.4f}  '
        f'p1={np.percentile(gc,1):.4f}  p99={np.percentile(gc,99):.4f}',
        f'α  mean={ac.mean():.4f}  std={ac.std():.4f}  '
        f'p1={np.percentile(ac,1):.4f}  p99={np.percentile(ac,99):.4f}',
    ]

    progress(0.85, desc='產生圖表與物理量轉換...')
    maps_fig = make_maps_figure(gamma_pred, alpha_pred)

    fr = float(frame_rate_hz) if frame_rate_hz and float(frame_rate_hz) > 0 else None
    time_map, D_alpha, time_lbl, d_unit, w_phys_nm = compute_physical_units(
        gamma_pred, alpha_pred, cell_mask, l_psf, pixel_size_eff_nm, fr)
    phys_fig = make_physical_figure(gamma_pred, time_map, D_alpha, time_lbl, d_unit, l_psf, w_phys_nm)

    tdc, dac = time_map[cell_mask], D_alpha[cell_mask]
    lines += [
        '',
        f'PSF width: {l_psf:.2f}px = {w_phys_nm:.1f}nm   '
        f'(λ={wavelength_nm}nm, NA={na}, 有效像素={pixel_size_eff_nm:.1f}nm)',
        f'τ_D [{time_lbl}]  mean={np.nanmean(tdc):.4g}  median={np.nanmedian(tdc):.4g}',
        f'D_α [{d_unit}]    mean={np.nanmean(dac):.4g}  median={np.nanmedian(dac):.4g}',
    ]

    gt_fig = None
    if mat_file is not None:
        try:
            gt_gamma_r, stats = compare_with_gt(mat_file, gamma_pred, cell_mask)
            gt_fig = make_gt_figure(gt_gamma_r, gamma_pred, stats)
            lines += [
                '',
                '=== Model vs iSCORS GT (γ vs D, expect +) ===',
                f'  Pearson={stats["pearson"]:+.3f}  Spearman={stats["spearman"]:+.3f}  '
                f'(|Spearman| is the metric; checkerboard-CV γ is primary)  N={stats["n"]}',
                f'  raw MAE={stats["mae"]:.4f} (scale-mismatched — ignore)  '
                f'z-scored MAE={stats["mae_z"]:.4f}',
            ]
            tifffile.imwrite(os.path.join(work_dir, 'gt_gamma_aligned.tif'),
                             np.nan_to_num(gt_gamma_r).astype(np.float32))
        except Exception as e:
            lines += ['', f'⚠ GT 比對失敗: {e}']

    progress(0.95, desc='儲存檔案並打包 zip...')
    tifffile.imwrite(os.path.join(work_dir, 'model_gamma.tif'), np.nan_to_num(gamma_pred).astype(np.float32))
    tifffile.imwrite(os.path.join(work_dir, 'model_alpha.tif'), np.nan_to_num(alpha_pred).astype(np.float32))
    tifffile.imwrite(os.path.join(work_dir, 'tau_D.tif'),       np.nan_to_num(time_map).astype(np.float32))
    tifffile.imwrite(os.path.join(work_dir, 'D_alpha.tif'),     np.nan_to_num(D_alpha).astype(np.float32))
    maps_fig.savefig(os.path.join(work_dir, 'model_maps.png'), dpi=120, bbox_inches='tight')
    phys_fig.savefig(os.path.join(work_dir, 'physical_units.png'), dpi=120, bbox_inches='tight')
    if gt_fig is not None:
        gt_fig.savefig(os.path.join(work_dir, 'gt_compare.png'), dpi=120, bbox_inches='tight')

    zip_path = os.path.join(work_dir, 'iscors_app_results.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(work_dir):
            if fn == 'iscors_app_results.zip':
                continue
            zf.write(os.path.join(work_dir, fn), arcname=fn)

    progress(1.0, desc='完成')
    return maps_fig, phys_fig, gt_fig, '\n'.join(lines), zip_path


# ───────────────────────────── UI ───────────────────────────────────────────
with gr.Blocks(title='iSCORS-Net Inference') as demo:
    gr.Markdown(
        '# iSCORS-Net 推論工具\n'
        '上傳 iSCAT 顯微影片 (`.tif`) 與訓練好的模型權重 (`.pth`)，'
        '推論每個像素的異常擴散參數 **γ**（速率）與 **α**（指數），'
        '並可選擇上傳 iSCORS MATLAB 結果 (`.mat`) 進行 GT 比對與物理單位換算。\n\n'
        '> 模型本身輸出無因次的 (γ, α)；物理單位 τ_D / D_α 是後處理（post-hoc）換算，'
        '同一份權重可用於不同光學設定的影片，只要在「光學與物理參數」中正確填入該影片的設定即可。'
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_file = gr.File(label='影片 (.tif)', file_types=['.tif', '.tiff'], type='filepath')
            ckpt_file  = gr.File(label='模型權重 (.pth)', file_types=['.pth', '.pt'], type='filepath')
            mat_file   = gr.File(label='[選用] iSCORS GT (.mat)', file_types=['.mat'], type='filepath')

            with gr.Accordion('前處理 / 模型設定', open=False):
                recon_taus_str = gr.Textbox(
                    label='RECON_TAUS（τ 清單，以逗號分隔，需與訓練時一致）',
                    value='1, 2, 4, 8, 16, 32, 48, 64, 96, 128')
                bin_factor  = gr.Slider(label='BIN_FACTOR（空間合併倍率）',
                                        minimum=1, maximum=8, step=1, value=2)
                n_frames    = gr.Number(label='N_FRAMES（讀取影格數上限）', value=2000, precision=0)
                gamma_scale = gr.Number(label='GAMMA_SCALE（γ 輸出上限，需與訓練時一致）', value=2.0)
                use_sigma   = gr.Checkbox(label='USE_SIGMA（模型是否使用 σ_G_norm 輸入通道，需與訓練時一致）',
                                          value=True)
                n_components = gr.Dropdown(label='N_COMPONENTS（前向模型成分數，需與訓練時一致）',
                                           choices=[1, 2], value=1)
                alpha_mode   = gr.Dropdown(
                    label='α 模式（N_COMPONENTS=2，需與訓練時一致）',
                    choices=['free (per-pixel)', 'fixed α=1', 'global α (shared scalar)'],
                    value='free (per-pixel)')

            with gr.Accordion('光學與物理參數（用於 τ_D / D_α 換算）', open=False):
                wavelength_nm = gr.Number(label='WAVELENGTH_NM（激發波長, nm）', value=532.0)
                na            = gr.Number(label='NA（數值孔徑）', value=1.4)
                pixel_size_nm = gr.Number(label='相機原始像素大小（nm/px，合併前；會自動乘上 BIN_FACTOR）',
                                          value=65.0)
                frame_rate_hz = gr.Number(label='FRAME_RATE_HZ（留空或 0 = 維持 frame-lag / pixel 單位）',
                                          value=0)

            run_btn = gr.Button('執行推論', variant='primary')

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab('γ / α 推論圖'):
                    maps_plot = gr.Plot(label='Model γ / α maps')
                with gr.Tab('物理單位換算'):
                    phys_plot = gr.Plot(label='τ_D / D_α')
                with gr.Tab('GT 比對'):
                    gt_plot = gr.Plot(label='Model vs iSCORS GT')
            stats_box = gr.Textbox(label='統計摘要', lines=10)
            zip_out   = gr.File(label='下載結果（TIF + PNG，打包為 .zip）')

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_file, ckpt_file, mat_file,
                recon_taus_str, bin_factor, n_frames, gamma_scale, use_sigma,
                n_components, alpha_mode, wavelength_nm, na, pixel_size_nm, frame_rate_hz],
        outputs=[maps_plot, phys_plot, gt_plot, stats_box, zip_out],
    )

    gr.Markdown(
        '---\n'
        '**注意事項**\n'
        '- 影片前處理流程（空間合併 → flat-field → 逐幀高斯背景去除）與訓練 notebook 完全相同；'
        'BIN_FACTOR / RECON_TAUS / GAMMA_SCALE / USE_SIGMA / N_COMPONENTS 必須與該權重訓練時的設定一致，'
        '否則推論結果無意義，甚至會因張量形狀不符而報錯。N_COMPONENTS=2 時模型輸出 '
        '[f, γ_slow, γ_fast, α]，會自動換算為有效 γ = f·γ_fast+(1-f)·γ_slow 後顯示。\n'
        '- GT 比對假設 `.mat` 內含 MATLAB iSCORS 的 `D_map`（或同義欄位）。'
        'γ 是 G(τ)=1/(1+γτ^α) 的衰減率，γ ∝ D，故與 `D_map` 直接比較（預期正相關）；'
        '舊版錯誤地反轉成 `1/D` 導致相關係數變負。iSCORS GT 沒有真實的 α，因此不比對 α。\n'
        '- τ_D、D_α 為後處理物理單位換算，模型本身維持無因次空間，'
        '同一份權重可重複用於不同光學設定的影片。'
    )

if __name__ == '__main__':
    demo.queue().launch()
