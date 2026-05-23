# iSCORS-Net

Physics-Informed Internal Learning for iSCAT diffusion parameter mapping.

Given a single iSCAT video (T × H × W), the network learns to output a dense map of diffusion parameters — **Gamma** (diffusion coefficient) and **Alpha** (anomalous exponent) — at every pixel, supervised only by a sparse 2% subset computed via traditional autocorrelation fitting.

---

## 方法概覽

| 元件 | 說明 |
|---|---|
| **Input** | 8-channel tau slice stack，從影片中按時間延遲取幀 |
| **Model** | `PISSLTauEncoder` — U-Net with PixelShuffle upsampling |
| **Output** | 2-channel map (Gamma, Alpha) per pixel |
| **GT 計算** | FFT + Hann window 自相關 → curve fitting，預快取 |
| **監督策略** | 隨機 2% 像素，每個 patch 只算中央 3×3 的 masked loss |
| **Random tau** | 每個 training step 隨機抽 8 個 tau 值，增加泛化 |

### 為什麼用 masked loss（最關鍵）

GT 只在中心像素 (y, x) 計算。若對整個 64×64 patch 算 loss，邊緣像素的 GT 是錯的（用中心值填充），會反效果。Mask 確保只有中央 3×3 有梯度。

---

## 專案結構

```
iscors-net/
├── train_internal.py          # 主要訓練腳本 (Internal Learning)
├── iscors_fast_runner.ipynb   # Colab 一鍵執行 notebook
├── params.json                # 光學與噪聲參數
├── datasets/
│   └── tau_sparse_dataset.py  # Tau 採樣 + GT 預快取 + center 3x3 mask
├── models/
│   └── pissl_tau_encoder.py   # U-Net (PISSLTauEncoder)
├── loss/
│   └── physics_loss.py        # Masked MSE loss
├── utils/
│   └── traditional_iscors.py  # FFT + Hann window 自相關 + curve fitting
├── data/
│   └── test_synthetic_cell.tif
├── checkpoint/
│   └── pissl_internal_v2.0.pth
└── result/
    ├── loss_curve_v2.0.png
    └── inference_maps_v2.0.png
```

---

## 在 Google Colab 執行

開啟 `iscors_fast_runner.ipynb`（選 `claude/beautiful-volta-gFUot` 分支），按順序執行：

1. **Version** — 設定版本號
2. **Setup** — `git clone` repo + `pip install`
3. **Video** — 生成測試影片（若不存在）
4. **Train** — `python train_internal.py`
5. **Results** — inline 顯示結果圖
6. **Download** — 下載 `results_v2.0.zip`

> 不需要手動上傳任何檔案。

---

## 本機執行

```bash
pip install -r requirements.txt
python utils/generate_test_video.py   # 建立測試影片
python train_internal.py              # 訓練
```

---

## 超參數 (v2.0)

| 參數 | 值 | 說明 |
|---|---|---|
| `TRAIN_RATIO` | 0.02 | 2% 稀疏監督像素 |
| `RANDOM_TAU` | True | 每 step 隨機抽 tau |
| `NUM_TAU_CH` | 8 | 輸入 channel 數 |
| `MAX_TAU` | 64 | 最大時間延遲（幀） |
| `PATCH_SIZE` | 64 | 空間 patch 大小 |
| `EPOCHS` | 30 | 訓練輪數 |
| `LEARNING_RATE` | 1e-4 | Adam |

---

## 版本紀錄

| 版本 | 變更 |
|---|---|
| **v2.1** | 修正 3 個訓練失敗根本原因（GT 過濾、輸入 normalization、bilinear 取代 PixelShuffle、Gamma bounded） |
| **v2.0** | FFT+Hann 自相關、隨機 tau、center 3×3 masked loss、train_ratio=0.02 |
| v1.x | 舊 zip 上傳流程、固定 tau、無 mask（已廢棄） |

---

## 模擬影片設計說明

`utils/generate_test_video.py` 生成三個擴散區域 + 靜態背景：

| 區域 | Gamma | Alpha | 說明 |
|---|---|---|---|
| 細胞本體（大圓） | 0.1 | 1.0 | 正常擴散 |
| 細胞器 1（左上小圓） | 0.5 | 1.5 | 快速異常擴散 |
| 細胞器 2（右下小圓） | 0.05 | 0.5 | 慢速次擴散 |
| 背景 | 0.0 | 0.0 | 近靜態（std=0.05）|

### 為什麼背景要近靜態（std=0.05）

舊版：`gamma=0` → `generate_signal_with_autocorr` 返回白噪音，variance=10，CV≈3.2%，與細胞像素相同 → curve fitting 跑出亂的 gamma 值（可能很大），汙染 GT。

新版：背景 std=0.05，CV=0.05% << 細胞 CV 3.2%。`fit_physical_parameters` 用 CV < 0.5% 作為靜態判斷，直接返回 (0,0)，不跑 fitting。

### 為什麼 T=500（舊版 T=200）

`max_tau=64` 需要計算到 G(tau=64)，每個 lag 的統計樣本數 = T - tau。
- T=200：最大 lag 的樣本數 = 200-64 = **136**（統計不足）
- T=500：最大 lag 的樣本數 = 500-64 = **436**（穩定）
