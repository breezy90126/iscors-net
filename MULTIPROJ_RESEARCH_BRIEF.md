# Research Brief: Multi-Projection Spatiotemporal Analysis from a Single iSCAT Video

## 背景脈絡

這個討論從 iSCORS-net 延伸出來。iSCORS-net 是一個 Physics-Informed Self-Supervised Learning 系統,從單一 iSCAT 影片用稀疏 τ 採樣訓練 U-Net 預測每像素的 (γ, α) 擴散參數。核心架構:

- 輸入: (B, 2K, P, P) — K 個稀疏 τ lag 的 G_norm(τ_k; y,x) + K 個 τ-PE channel
- 輸出: γ_map, α_map — 對應廣義擴散係數與異常擴散指數
- 自監督: physics loss `L = Σ_k w_k · (G_theory_norm(τ_k; γ,α) − G_norm(τ_k))²`
- 無需 ground truth,完全從影片自身的時空結構學習

**這個 brief 的起點**:per-pixel ACF 是 (T,H,W) 原始影片的一種有損投影。問題是:存在哪些資訊論意義上不同、且攜帶了 ACF 丟棄資訊的互補測量?它們能從同一支影片工程榨取嗎?

---

## 核心主張

一支 (T, H, W) iSCAT 影片包含的資訊可以用四個互補投影來覆蓋,每個投影回答不同的物理問題,且每個投影丟棄不同的東西:

### 投影 1: Per-pixel ACF → G(τ; y,x)
```
G(τ; y,x) = <δI(y,x,t) · δI(y,x,t+τ)> / <I(y,x)>²
```
- **回答**: 這個像素的動態怎麼在時間上衰減?
- **提取**: (μ_D, σ_D) ≡ (γ, α) — 每像素擴散分布的均值與寬度
- **丟棄**: 所有跨像素關係、方向性、空間結構
- **根本限制**: σ_D 估計病態(Laplace inversion 奇異值指數衰減)

### 投影 2: DDM → f(q, τ)
```
f(q, τ) = <|FFT(I(t+τ) − I(t))|²>_t
```
- **回答**: 這個動態系統是哪種運動型態?
- **提取**: Γ(q) scaling — q² → 擴散; q → 有向流; plateau → 受限; fractional → 異常
- **丟棄**: 空間解析度(壓成 q-magnitude)
- **獨特貢獻**: 不假設運動模型,從數據驗證 ACF 的物理假設是否成立
- **隱含尺度**: q* 的倒數給出受限運動的特徵尺寸

### 投影 3: STICS → r(Δy, Δx, τ; y,x)
```
r(Δy,Δx,τ; y,x) = <δI(y,x,t) · δI(y+Δy,x+Δx,t+τ)> / (<I(y,x)>·<I(y+Δy,x+Δx)>)
```
- **回答**: 鄰近像素的動態有時空關聯嗎?有定向流嗎?
- **提取**: v(y,x) 速度場;σ_D 的獨立空間側量(spatial profile 的非高斯性)
- **與 ACF 的關係**: (Δy,Δx)=(0,0) 退化為 ACF — ACF 是 STICS 的特殊情況
- **關鍵物理**: 有向輸運 vs 被動擴散可被區分(ACF 裡兩者幾乎無法分開)

物理模型:
```
r(Δ,τ; γ,α,v_y,v_x) ∝ exp(−((Δy−v_y·τ)²+(Δx−v_x·τ)²)/(4·D_eff·τ)) · G_norm(τ; γ,α)
```

σ_D 的第二個測量軸:
```
r(Δ,τ) = ∫ p(D) · (4πDτ)⁻¹ · exp(−|Δ|²/(4Dτ)) dD
```
p(D) 寬 → spatial profile 非高斯(尾巴厚);p(D) 窄 → 接近高斯。
這比 Laplace inversion 更穩定,是 σ_D 估計的根本改善。

### 投影 4: χ4 → 四點動態異質性
```
χ4(τ) = Var_space(C(y,x,τ))
其中 C(y,x,τ) = local mobility/correlation at pixel (y,x) and lag τ
```
- **回答**: 動態異質性(σ_D 大)是靜態的還是動態的?
- **靜態異質性 (quenched disorder)**: 不同區域永遠有不同動態 → 反映細胞的永久結構
- **動態異質性**: 同一區域在快/慢態之間切換,χ4 在 τ* 有峰值 → LLPS 的動力學特徵
- **獨特貢獻**: 唯一能區分這兩種情況的量測;對 σ_D 的物理詮釋有決定性影響

---

## 四個投影的互相約束網絡

這四個投影不是獨立的——它們形成一個一致性網絡:

```
DDM Γ(q) scaling
    ↕ 驗證 ACF 的運動模型假設
ACF (μ_D, σ_D)
    ↕ σ_D 獨立驗證(spatial non-Gaussianity)
STICS r(Δ,τ)  ──→ 修正 v 對 ACF γ 的污染
    ↕ 靜態/動態異質性詮釋
χ4 ξ4(τ), τ*
```

**一致性交叉檢查(關鍵)**:
- DDM 說 Γ(q) ∝ q¹(有向流)但 STICS 說 v≈0 → 矛盾 → 系統誤差
- ACF 說 σ_D 大但 STICS spatial profile 是高斯 → ACF 的 σ_D 是噪聲,不是真實異質性
- STICS 偵測到 v > 0 的區域 → ACF 在該區域的 μ_D 被污染,需修正
- DDM 說受限運動,q* ≈ 1/200nm → χ4 說 dynamic heterogeneity,τ* = 30s → 兩者共同支持 LLPS 解釋

---

## 設計哲學:知道自己的解析力邊界

一個設計良好的框架目標不是「永遠給出更細緻的答案」,而是知道自己解析力的邊界在哪裡,並在邊界外誠實地降級。

每個投影有各自的失效條件:

| 投影 | 失效條件 | 誠實降級 |
|---|---|---|
| ACF μ_D | 低 T、低 SNR | 可靠,最後失效 |
| ACF σ_D | 低 T、多成分混合 | 報告置信區間;SNR 不足時標記不可信 |
| DDM Γ(q) | 高 q 被散粒噪聲淹沒 | 無法確定特徵尺寸;只報告低 q 行為 |
| STICS v(y,x) | 慢漂移低於解析力 | 無法排除非常慢的主動運輸 |
| χ4 | 最需要高 SNR、長 T | 最先失效;無法判斷 LLPS vs 固定異質性 |

降級順序: χ4 → DDM(高 q) → STICS v → σ_D → μ_D

框架輸出不只是四張地圖,而是:
1. 每個參數的信賴區間(σ_G 已在 iSCORS-net 的 `sn2n_sampling.py` 實作)
2. 一致性交叉檢查結果(有/無矛盾)
3. 明確標記哪些問題超出當前 data 的解析力

---

## 具體應用範例:iSCAT 染色質濃縮

```
DDM:   Γ(q) plateau at q* ↔ 濃縮域尺寸 ~200 nm
ACF:   μ_D 內部 0.05 μm²/s,外部 0.3 μm²/s;內部 α = 0.7(次擴散)
STICS: v ≈ 0 in interphase → 被動過程(非馬達驅動)
χ4:   τ* = 30s dynamic heterogeneity → LLPS 行為
```

整合詮釋:「染色質進行液-液相分離,形成 ~200nm 動態液滴。內部運動受限且黏彈性。被動熱力學驅動,材料交換時間 ~30s。」

每一句話來自不同投影,且互相約束:LLPS 需要 dynamic χ4 + DDM 受限同時成立;「被動」需要 STICS v≈0 支持。

---

## 文獻缺口

目前最接近的工作:

- **Scipioni et al. 2018, Nature Communications** — Comprehensive Correlation Analysis (CCA):在同一份數據上同時跑多種關聯方法。**但**: 需要 Zeiss Airyscan 32-element detector;螢光顯微鏡;沒有 χ4;沒有 resolution-limit awareness。

- **Di Rienzo et al. iMSD** — ACF + STICS 組合,從單一影片提取 diffusion/flow/confinement。**但**: 沒有 DDM q-scaling 驗證,沒有 χ4。

- **"Hierarchical Heterogeneities in Spatio-Temporal Dynamics of the Cytoplasm" (biorxiv 2025)** — 用 DDM 提取非高斯性(≈ σ_D 分布)。**但**: 只用 DDM 一個投影,cell-free 系統,沒有 spatial map。

**不存在的**:
- ACF + DDM + STICS + χ4 全部從同一支普通單相機影片
- 套用在 iSCAT(非螢光,不同物理)
- 帶有明確 resolution-limit awareness 和 honest degradation
- ML 加速版本(稀疏 τ 採樣 → 網路推斷)

---

## 與 iSCORS-net 的關係

iSCORS-net 的 τ 稀疏採樣就是 FAST 精神的體現:

| | FAST (fluorescence denoising) | iSCORS-net |
|---|---|---|
| 稀疏採樣的對象 | 特定 frame pairs at τ₁, τ₂... | G(τ₁), G(τ₂),...,G(τ_K) |
| 堆成 channel 餵入 2D 網路 | 是 | 是 |
| 自監督 | 預測另一 sub-sample | G_theory vs G_empirical |

這個 brief 的問題是:如果在 iSCORS-net 的輸入層加入 STICS channels r(Δ,τ_k),並設計對應的 physics loss,同時保留原有的 ACF channels,能在不大改架構的前提下讓網路直接輸出 (γ, α, v_y, v_x)?DDM 和 χ4 應該在網路輸入層還是作為後處理的交叉驗證層?

---

## 開放討論問題

1. **架構整合順序**: DDM 和 χ4 是否應該成為額外的 input channel(讓網路從多投影聯合推斷),還是在推斷之後作為 consistency check?

2. **STICS 的 physics loss**: `r_theory(Δ,τ; γ,α,v_y,v_x)` 在多成分系統(多種 D 共存)下是否仍然可寫成解析式?還是需要 numerical integration?

3. **自監督的配對**: STICS channels 的 blind-spot 應該怎麼設計?對一個空間位移 (Δy,Δx) 做 masking 是否等效於現有的 per-pixel blind-spot?

4. **Honest degradation 的實作**: 信賴區間和 consistency flags 應該是網路輸出的一部分(learned uncertainty),還是基於 σ_G 的解析估計?

5. **χ4 的計算成本**: χ4 需要最高 SNR 和最長 T,是否值得作為網路輸入,還是只作為 post-hoc 驗證?

6. **iSCAT 的特殊性**: iSCAT 的訊號是干涉散射(非螢光 Poisson noise),這個框架的噪聲模型需要如何調整才能正確估計各投影的信賴區間?
