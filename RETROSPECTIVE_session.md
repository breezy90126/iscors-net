# iSCORS-Net — Session Retrospective

100 個 commit。這個 session 的開發史，分成三條線總結。

## 一、科學探討線（從質疑 slope=3 → 找可複現軸）

起點是質疑 iSCORS 論文的 **slope=3**，一路推進：

1. **上傳論文 PDF** + 復刻 condensation（slope-3 投影、y-intercept）。
2. **用手畫 mask** 分割細胞核（Otsu/GMM 失敗 → 改讀你的 `condensation_mask.tif`，並修正「內部區＝核」的極性）。
3. **PCA vs slope-3**：發現核內 per-pixel 斜率 ~0.5，**不是 3**——建立「within-cell ≠ cross-cell」的關鍵區分。
4. **可複現性框架**（half-split）：發現「PC1=95% 變異」其實大半是 **X、Y 共享雜訊**；真正該最大化的是**可複現性**（廣義特徵值 `Csig w = λ Ctot w`），不是變異。
5. **共模去噪**：X⊥Y、reliability-max 軸比原始通道更乾淨，因為**兩通道一起做能抵消共模雜訊**——這是「為什麼需要第二維」的量化答案。
6. **full-τ GEVD**：把整條 ACF 當特徵 → 找到 (γ,CV²) 丟掉的多條形狀軸；但逐條驗證後，**穩健的生物軸其實只有 ~1 條（核仁/密度）**，其餘是**光學梯度 + speckle + 雜訊**（用 nuisance 穩健性、drift 測試、landscape 峰數一一排除）。
7. **一路的誠實更正**：多次過早下結論（「只有一條軸」「CV² 就夠」「axis3 是 α」），都被你的追問和資料推翻並修正。
8. **物理落點**：可複現軸校準不了 slope=3（跨核律）；LLPS 要看 bulk 雙峰（結果 bulk 單峰）；要真正的新軸只能**換採集**（長時程/相位/螢光 marker）或加 **OT** 這種正交量測。

## 二、演算法貢獻線（可重現的分解）

- **reliability-max 廣義特徵分解**（取代變異-PCA）。
- **cross-half 訊號協方差** + **多次打亂 95% 雜訊地板**。
- **de-nuisance**（poly3+radial）分離光學平滑場。
- **FastICA + kurtosis 排序 + 固定符號**：讓 comp1 恆為稀疏成分、跨版本可重現（解決「要手挑 axis 2/4」的問題）。
- **Fourier 方向 band-stop 去斜條紋**（Section 3b，驗證 r∈[60,110], 40°）。
- **最新**：核仁恆為藍色（用核仁 mask 定符號）、線性 de-ramp 針對慢軸 comp2。

## 三、工程/交付線（RAM + notebook 整理）

這條走了不少冤枉路，最後收斂：

- 把爆炸的探索 notebook **整理成精簡三段式**（復刻論文 / 單細胞最重複軸 / 多軸 GEVD-ICA + 比較圖 + viewer），移除 LLPS 和重複載入。
- **RAM 大迷航**：引入 `streaming_acf`（memmap 惰性讀）本意省 RAM，但反而弄壞；真相是 full-load（分塊讀+bin 成 2GB）本來就夠，之前爆 RAM 是十幾格各自重載，不是單次載入。最後改回 full-load + CACHE 架構。（`streaming_acf` 留在 utils 備用。）
- **gradio** 收成單圖 + 下拉選單。
- 加了 **GitHub bootstrap**、**config**、`condensation_mask.tif` 進 repo、`streaming_acf` 進 `utils`。

## 一句話總結

這個 session 把「iSCORS 單核資料到底能可複現地分出幾條軸」這個問題做到底：答案是**穩健生物軸大約 1 條（核仁/密度）**，共模去噪讓它比單通道乾淨；slope=3 是跨核律、within-cell 站不住；要更多獨立軸只能換採集。過程中建立了一套可重現的分解流程（廣義 GEVD + kurtosis-ICA + 去 nuisance/條紋 + half-split 驗證），並把探索 notebook 收斂成一份能跑、精簡、記憶體安全的 deliverable。

核心檔案：`iscors_deliverable.ipynb`（重寫多次）、`utils/gpu_iscors_fit.py`（加 `streaming_acf`）、`papers/PMC11196589.pdf`、`data/condensation_mask.tif`。
