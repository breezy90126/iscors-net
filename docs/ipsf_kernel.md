# 用模擬 iPSF 建 kernel，把 G(τ) 反演成 P(D)

接續 `docs/laplace_projection_alpha.md`。那份文件的第一節指出：用 DLS 的指數 kernel
`e^(−Γτ)` 去反演 iSCAT 的 per-pixel ACF 是 kernel mismatch——一個 pixel 收的是 iPSF
涵蓋的整段 q。這裡把那個 q 權重**直接從 iPSF 算出來**，放進 kernel，反演出來的就是 P(D)。

程式：`utils/ipsf_kernel.py`（純 numpy）。驗證：`python3 scripts/validate_ipsf_kernel.py`（約 1 分鐘）。

---

## 一、kernel 是什麼

粒子在 xy 平面以 D 擴散（彼此獨立）。由 Wiener–Khinchin，binned pixel 的強度自協方差是

```
C(τ; D) = Σ_q S(q) · exp(−D |q|² τ Δt),     S(q) = |FT[iPSF ⊗ pixel box](q)|²,  q ≠ 0
```

pipeline 的 `nor1` 估計量是 Ĉ(τ)/Ĉ(0)，所以 kernel 的一欄是

```
K[τ_k, D_j] = C(τ_k; D_j) / C(0; D_j)
```

每一欄都是「由光學決定權重的指數和」，所以 iPSF 的 q 積分已經在 kernel 裡，
不會再被反演誤認成多分散。

**finite-T 偏差直接放進 kernel**（`ipsf_kernel(..., T=N_FRAMES)`）：
不去修資料，而是讓每一欄等於 pipeline 估計量（扣樣本平均、除以 Ĉ(0)）的**期望值**。
公式是精確的（`finite_t_expected`）：

```
E[Ĉ(τ)] = C(τ) − (1/n) Σ_{t<n} (a_t + a_{t+τ}) + Var(x̄),   a_t = Cov(x_t, x̄),  n = T − τ
```

這正是 `laplace_projection_alpha.md` 6.4 第 3 點「解析修正」的正向模型版本，而且不用迭代。

## 二、iPSF 從哪來

| 來源 | 用途 |
|---|---|
| `scripts/export_ipsf_stack.py` | 用帶回來的完整向量模擬器 `datasets/xyt_dataset_generator.py`（極化、Rayleigh/Mie、相位遮罩）輸出單粒子 iPSF 堆疊 `(nz, H, W)`。**需要 torch，在 Colab 跑。** |
| `scalar_ipsf()` | 純 numpy 的純量 pupil 模型（有環紋與 z 相位翻轉，沒有極化與超臨界角）。驗證用，也可以在沒有 torch 時先頂著。 |

`ipsf_spectrum(stack, px, bin_factor, z_weights)` 對各 z 的 S(q) 加權平均——
代表**靜態**的高度分佈（每顆粒子在 lag 內不換 z）。

## 三、驗證結果（`scripts/validate_ipsf_kernel.py`）

### 3.1 kernel 對上暴力模擬

粒子在週期性盒子裡 2D 布朗運動、通過細取樣的 iPSF，記錄一個 binned pixel T=2000 幀，
300 個獨立 pixel 平均：

| 情況 | finite-T kernel 的 mean z² | 理想 kernel（不含偏差）|
|---|---|---|
| 快（D q²Δt = 0.3） | **1.04** | 4.92 |
| 慢（D q²Δt = 0.03） | **0.33** | 32.7 |

z² ≈ 1 表示跟模擬在統計誤差內一致。不含 finite-T 偏差的理想 kernel 在慢的情況下
偏離到 32 倍誤差：τ=128 時理想值 0.177，實際只有 0.089。

**一個要改的地方：ROI 平均要用「比值的平均」還是「平均的比值」。**
kernel 描述的是 E[Ĉ(τ)]/E[Ĉ(0)]。若先算每個 pixel 的 g_norm 再平均（現在 notebook 的做法），
當 Ĉ(0) 本身漲落大（訊號稀疏、間歇）時會多出一個 Jensen 偏差——
上表慢的情況 τ=128 會從 0.089 再掉到 0.030。解法很便宜：

```
g_roi(τ) = Σ_ROI g_norm(τ)·c0  /  Σ_ROI c0        # c0 = compute_g_norm_torch 回傳的 g_zero
```

### 3.2 單一 D：iPSF kernel vs 指數 kernel

同一條單一 D 的曲線：

- iPSF kernel：**一個峰**，落在真值（3.16e-16 vs 3.20e-16），寬 0.09 decade；
- 指數 kernel：**兩個峰**，寬 0.38 decade——q 積分被誤認成多分散，正是第一節說的 mismatch。

### 3.3 兩群混合（50/50），ROI 級雜訊，各 20 次雜訊抽樣

| D 比 R | 雜訊 0.13%（~1000 px） | 雜訊 0.04% |
|---|---|---|
| 2 | 0/20 | 0/20 |
| 3 | 16/20 | 20/20 |
| 5 | 20/20 | 20/20 |
| 10 | 20/20 | 20/20 |
| 30 | 20/20 | 20/20 |

（「分出來」= 兩個峰、各自離真值 0.15 decade 以內。）

**這張表是樂觀上限**：資料和反演用同一個 kernel 產生（inverse crime），雜訊是各 τ 獨立的白雜訊
（真實的 Ĝ(τ) 雜訊在 τ 之間是相關的）。下一步要用模擬器產生的完整影片走一次真實 pipeline 再測。

## 四、反演：`invert_kernel`

CONTIN 式：`min ||W(Kp − g)||² + λ||L p||²，p ≥ 0`，L 是二階差分，用 FISTA 解。
跟原本的 Richardson–Lucy `contin()` 不同，它不要求 K、g 為正——finite-T kernel 在長 τ 會微負。

`g0_anchor=1.0`（預設開）：`nor1` 在 τ=0 恆為 1，每欄 kernel 在 τ=0 也是 1，
所以加一列 Σp = 1 是免費的資訊。沒有它，solver 會把質量塞到「一幀內就衰減完」的最快 D 上，
在 grid 邊緣做出假峰（等效一個常數偏移）。有了它，p 直接就是族群比例；
若真的有比一幀還快的族群，會以快端邊緣的質量出現。

λ 的選法：讓殘差 RMS 落在 half-split 雜訊附近（同 `roi_spectrum` 的標準）。

## 五、還沒做的 / 限制

1. **z 方向運動**沒有進 kernel（只有靜態 z 分佈）。z 改變干涉相位，對快的軸向運動要用
   模擬器做完整 3D 模擬來檢查 kernel。
2. 已接進 `iscors_one_shot_diagnostics.ipynb` 的 **D6**（由 `scripts/make_one_shot_notebook.py` 產生）：
   c0 加權 ROI 曲線 → `ipsf_kernel(..., T)`（全段與半段各一個 kernel）→ `invert_kernel`
   → half-split 檢查峰是否重現，P(D) 以 µm²/s 畫出，並標上論文的 D* = 0.92 µm²/s。
   實驗參數取自論文 Methods（Hsiao et al., Commun. Biol. 7:763, 2024）：
   **COBRI（穿透式 iSCAT）、660 nm、OBJ2 NA 1.49、42 nm/px（640×640）、5000 fps**。
   注意舊設定裡的 520/532 nm、65 nm/px 跟論文不符，已在 notebook config 改正波長與幀率。
3. 論文本身用的是單一 k 的 DLS 式 `ACF = exp(−D*k²τ)`，k 用奈米粒子校正。
   D6 的 iPSF kernel 是另一個假設（橫向 q 積分）；兩者給的 D 可以互相對照，
   但 k 的校正值只有在 SI 裡，D6 的絕對值目前取決於 iPSF 模型（純量模型只是近似）。
4. 染色質不是稀薄粒子：連續介質的情況下權重還要乘上未知的靜態結構因子 S_ρ(q)，
   目前假設它在通帶內是平的。
5. per-pixel 仍然不可行（雜訊 4.1% 下沒有可分辨的第二群），這仍是 ROI 級的分析。
