# One-shot 診斷測試結果：Zenodo COBRI 影片（2026-11-21）
來源：`iscors_one_shot_diagnostics.ipynb`，Colab 執行 `one_shot_outputs/20261121_0405`（log.txt、results.json、圖檔）。
數字直接取自 log；「判定」欄為 notebook 自動 verdict，必要時附註說明。

## 0. 資料與參數

| 項目 | 值 |
|---|---|
| 影片 | `COBRI_rarw_video.tif`（4.1 GB），frames [0, 5000)，5000 fps（1 s），binned 320×320 |
| 核 mask | 38643 px（共 102400） |
| iPSF | **scalar pupil model**（未設 `IPSF_NPZ`）；有效 q = 9.02 rad/µm（1/q = 111 nm）；D ∝ 1/q²，q 錯 2× → D 錯 4× |
| 軸向 | q_z = 26.3 rad/µm（每 239 nm 軸向位移一個條紋） |
| CONTIN | iPSF kernel + 白噪聲 nugget + τ=0 錨點 + edge bucket；λ ∈ {1e‑4, 1e‑3, 1e‑2} |
| D6 雜訊篩選 | cross‑half GEVD，丟掉最不可重現的 30%（38643 → 27050 px）；`D6_FEATURES='G+CV2'` |
| 去漂移（D9/D11） | 置中 boxcar high‑pass，`HP_W=1501`，資料與 kernel 同步；有效 T_eff = 3500 |
| D10 tile | `TILE_PX=12`（268 tiles）；快/慢分界 D = 0.8 µm²/s |
| D11 | Wiener；VID_STEP=50、VID_SUB=10、VID_WIN=500 |

## 1. 各段結果

### D1 頻譜 / 白噪聲
- 白噪聲佔 per‑pixel 變異 **17.8%**（預處理後）；高頻斜率 −0.12（已到 white floor）。
- PSD 推得 α = 0.21，ACF 擬合 α = 0.48（兩者不一致，PSD 法受白噪聲底影響）。
- 判定：**無 aliasing**；無共模儀器譜線。

### D2 線性區
- 核內 kurtosis 中位數 2.88（高斯 = 3），< 2.5 的 pixel 4.3%。
- 判定：**LINEAR REGIME**，小相位近似成立。

### D3 ACF 形式
- ROI 平均曲線（38643 px），半段噪聲 0.00227。
- algebraic A/(1+γτ^α)：rms 0.00630（γ=0.0935, α=0.578, A=0.867）；stretched exp：rms 0.00689（α=0.398）。
- 判定：**INDISTINGUISHABLE**（差 0.00059 < 噪聲）；兩種形式都擬合不到噪聲水準（rms ≈ 2.8× 噪聲）。

### D4 / D4b 有限 T 偏差
- 全段 α = 0.476，半段 0.543 / 0.543（d = +0.067）；γ 依賴趨勢 slope +0.388，span 0.243 dec。
- D4 判定：**PRESENT AND GAMMA‑DEPENDENT**。
- D4b（exact expected estimator 正向擬合，γ 分 bin）：α 全段/半段 0.66/0.74,0.72 → **0.54/0.48,0.48**；
  bin span 0.130 → **0.054 dec**（排除最慢 bin），d_α +0.070 → −0.060。
- 修正後白噪聲比例（1−A）各 γ bin：0.02–0.12（與 D1 的 18% 同量級，但偏低）。
- 判定：**FINITE‑T EXPLAINS IT**。
- 注意：per‑pixel（非 bin）層級修正後 span 反而 0.277 → 0.459 dec，per‑pixel 擬合本身不穩（見 D3 討論，per‑pixel γ/α 欠定）。

### D5 ROI 的 γ 分布（per‑pixel 擬合的分布，非 CONTIN）
| ROI | 峰數 | centroid (log10 γ) | 寬度 | 1‑comp 殘差 / 噪聲 |
|---|---|---|---|---|
| whole nucleus | 1（半段 1/1） | −1.86 | 0.75 dec | 0.01551 / 0.00227（6.8×） |
| nucleoli | 1（半段 0/1） | −1.67 | 0.73 | 0.02103 / 0.00147 |
| nucleoplasm | 1 | −1.88 | 0.74 | 0.01412 / 0.00250 |
| whole nucleus [finite‑T 修正] | 1 | −2.03 | 0.80 | 0.00680 / 0.00130（5.2×） |
- 判定：pixel 之間 γ 的分布單峰（寬約 0.75 dec）；單一 ACF 擬合殘差約噪聲的 5–7 倍 → 單一成分不夠。

### D6 iPSF‑kernel CONTIN（核內）
- GEVD 可重現軸：G [0.78 0.71 …] keep 2；G+CV2 [0.82 0.72 0.70 …] keep 3；XY [0.82 0.14] keep 1。
- 保留 pixel 重疊（Jaccard）：G/G+CV2 0.76、G/XY 0.57、G+CV2/XY 0.65。
- filtered nucleus（27050 px，原始影片，未去漂移）：

| λ | white | slow‑edge | fast‑edge | 內部峰 D (µm²/s) | mass | resid |
|---|---|---|---|---|---|---|
| 1e‑4 | 0.10 | 0.05 | 0.06 | 0.125, 5.12, (381) | 0.36, 0.19, 0.01 | 0.00551 |
| 1e‑3 | 0.14 | 0.14 | 0.01 | 0.111, 5.12 | 0.27, 0.17 | 0.00553 |
| 1e‑2 | 0.16 | 0.19 | 0.00 | 0.078, 5.12 | 0.16, 0.15 | 0.00558 |

- 三種篩法（best λ）：

| 篩法 | white | 峰 D | mass |
|---|---|---|---|
| G kept / dropped | 0.10 / 0.06 | 0.125, 5.12 / 0.125, 4.03 (+19.1) | 0.36, 0.18 / 0.33, 0.19 (+0.02) |
| G+CV2 kept / dropped | 0.10 / 0.06 | 0.125, 5.12 / 0.111, 3.58 | 0.36, 0.19 / 0.32, 0.17 |
| XY kept / dropped | 0.09 / 0.07 | 0.125, 4.54 / 0.125, 3.58 | 0.36, 0.19 / 0.31, 0.17 |

- 判定（嚴格）：**NO model fits to the noise**（best resid 0.0055 vs 2× 噪聲 0.0041）→ notebook 只保證內部 centroid（0.63 µm²/s）的數量級。
  兩峰位置在 3 個 λ、3 種篩法、kept/dropped 之間一致，但殘差仍約 2.7× 噪聲，存在模型外系統誤差。
- GEVD 分群：group 1 判定 ONE broad distribution（峰分裂依賴 λ）；group 2、3 判定 systematics dominate。
- 末端（381 µm²/s）峰：fast‑edge 白噪聲 / 幀內去相關，非真實族群。

### D7 ICA ROI（finite‑T 修正，per‑ROI）
| ICA 分量 | ROI | px | γ_ft | α_ft | white | CONTIN 判定 |
|---|---|---|---|---|---|---|
| condensate +1 | puncta | 1496 | 0.0055 | 0.52 | 0.02 | 2 populations, stable |
| condensate +1 | large 1 / 2 | 80 / 61 | 0.0051 / 0.0087 | 0.50 / 0.42 | 0.02 | 2 peaks stable / systematics |
| — | nucleoli | 4585 | 0.1667 | 0.44 | 0.10 | systematics（centroid 1.42） |
| condensate +1 | background | 28709 | 0.0775 | 0.40 | 0.04 | ONE broad |
| other −1 | large 1 / 2（核仁塊） | 1076 / 454 | 0.0512 / 0.0737 | 0.26 | 0.04 | 2 peaks（約 0.1–0.18, 6.5–8.3） |
| other +1 | puncta | 53 | 0.0601 | 0.60 | 0.06 | 3 populations（小 ROI） |
- 核仁 γ_ft 約為背景的 **2.2×**（0.167 vs 0.078）→ 核仁較快。
- condensate puncta γ_ft 約為背景的 1/14 → 明顯較慢（但 G(1)=0.90、ROI 小，噪聲大）。
- 先前 condensate puncta 多為 ROI 邊緣假影，已加 `ICA_EDGE_PX=6` 排除。
- 半橢圓（other −1）推測為 perinucleolar 殼，**未驗證**。

### D8 每 pixel 兩群擬合（原始影片）
- D_fast = 5.12、D_slow = 0.125 µm²/s（比值 41×；去漂移後見 D9）。
- 快群比例 f：中位數 0.42，IQR 0.34–0.50，單峰；per‑pixel 兩群殘差中位數 0.0147。
- ROI 中位 f：puncta 0.22、large 1/2 0.20/0.28、nucleoli 0.48、background 0.43。
- 判定：**兩群在 pixel 內共存**（f 單峰）。

### D9 去漂移檢驗（HP_W = 1501）
| ROI | 慢群 mass | 慢群 centroid | 快群 centroid |
|---|---|---|---|
| filtered nucleus | 0.59 → 0.43（0.73×） | 0.136 → **0.255** | 11.8 → 17.7 |
| nucleoli | 0.45 → 0.33（0.73×） | 0.276 → 0.324 | 18.2 → 23.2 |
| background | 0.55 → 0.41（0.75×） | 0.146 → 0.262 | 15.7 → 18.3 |
- 無漂移模擬下 high‑pass 本身也讓慢群 mass 保留約 0.73× → 慢群存活。
- 此次 log 判定印出 MIXED（舊版判定邏輯）；新版（commit dfdfb7c）以 mass 存活判定 → **慢群為真，位置被漂移拉低**。
- 「快群 centroid」為 D > 0.8 全部質量的 centroid，含 fast‑edge，故大於快群峰位（5.1）。

### D10 Tile CONTIN（12×12 px，268 tiles）
- slow_D 中位 0.119（IQR 0.091–0.162）；fast_D 中位 16.7（IQR 14–20.7）；fast_share 中位 0.454（IQR 0.41–0.51）。
- 快群比例圖半段 r = **0.41**（< 0.5，空間圖案僅部分可重現）。
- 注意：tile 的 fast_D（16.7）高於 ROI 的快群峰（5.1）；tile 未去漂移、pixel 少，fast‑edge 漏入，**未解決**。

### D11 Wiener 分離（CONTIN P(D) + iPSF 導出）
- 使用族群：slow D ≈ 0.255、fast D ≈ 5.12 µm²/s（分界 1.14）；τ ≈ 242 / 12 frames；white 0.10。
- 預期頻帶純度：慢帶 91%、快帶 72%。
- 快群比例圖（3×3 band power unmixing，含 high‑pass 轉移函數）：中位 **0.40**，半段 r = **0.19**。
- 快帶與慢帶 RMS 圖為同一圖案（核仁暗）；快帶較平滑是因估計噪聲小。
- 影片：慢帶為 PSF 尺寸（2–3 px）speckle 原地去相關，無可追蹤的移動結構；兩支影片皆涵蓋同樣 0.7 s（去漂移後可用段）。

## 2. 圖檔對照
| 段落 | 檔名 |
|---|---|
| D1 | fig01_D1.png |
| D2 | fig02_D2.png |
| D3 | fig03_D3.png |
| D4 / D4b | fig04_D4.png、fig05_D4.png |
| D5 | fig06_D5.png |
| D6 | fig07_D6.png、fig08_D6.png、fig09_D6.png；D6b：fig23_D6b.png、fig24_D6b.png |
| D7 | fig10–12_D7.png、fig17–22_D7.png；`d7_roi_labels.npy` |
| D8 | fig13_D8.png；`d8_fast_fraction.npy` |
| D9 | fig14_D9.png、fig15_D9.png |
| D10 | fig16_D10.png；`d10_tile_maps.npz`；share_tiles.png、slow_tiles.png |
| D11 | d11_wiener_filters.png、d11_band_maps.png（fig25/26_D11.png）、d11_slow_band.mp4、d11_fast_share.mp4 |

## 3. 可信度整理

**可信（多重檢驗一致）**
- 小相位線性區成立（D2）；無 aliasing（D1）。
- 白噪聲不可忽略：約 10–18%（D1、D4b、D6 三種估計同量級）。
- D4 的 γ 依賴偏差來自有限 T，可被修正（D4b，bin 層級）。
- 單一 ACF 成分不足：殘差約噪聲 5–7 倍（D5）。
- 內部存在快（≈ 4–5 µm²/s）與慢（≈ 0.1–0.26 µm²/s）兩個峰：跨 λ、跨三種篩法、kept/dropped、ROI 一致；去漂移後慢群仍在。
- 兩群在 pixel 內共存，快群比例約 0.4 且空間上近乎均勻（D8 單峰、D10 r = 0.41、D11 r = 0.19）。
- 核仁比核質快（γ_ft 約 2.2×）。

**未定 / 需注意**
- **絕對 D**：scalar iPSF；需實測 iPSF（`IPSF_NPZ`）。
- **CONTIN 嚴格判定多為 "systematics dominate"**：最佳殘差仍約 2–3× 噪聲，兩峰「位置」穩定，但模型外仍有系統誤差（可能是 iPSF 形狀、共模雜訊、殘留漂移）。兩群為現有證據下最簡單的描述，非嚴格證明。
- 慢群的確切位置：原始 0.125 → 去漂移 0.255；快慢比因此從 41× 變 20×。
- D10 tile 的 fast_D（16.7）與 ROI 峰（5.1）不一致。
- α 的物理意義：ACF 形狀的有效指數（含 iPSF 多 q 平均），不是 MSD 指數。
- D7：condensate puncta 較慢、半橢圓為核仁殼，皆為單一影片、小 ROI 結果，未驗證。
- 兩群的生物學身分（染色質快/慢模態 vs 流動池/束縛池）為假說。
- 僅單一細胞、單一影片；無多細胞統計、無擾動實驗。
- 共模雜訊尚未扣除（下一版 TODO 1），可能是殘差高於噪聲的原因之一。
