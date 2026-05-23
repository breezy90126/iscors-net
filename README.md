# Spiral iSCAT + FAST-like XYZ 座標 MVP 專案規格說明 (Spec)

這個規格說明 (Spec) 是為了讓 AI 輔助程式碼編輯器如 **Cursor**（或類似工具，如 VS Code with GitHub Copilot）能夠理解整個專案的來龍去脈，並引導你逐步實現 MVP (Minimum Viable Product)。Spec 基於我們的對話脈絡：從傳統逐幀 fitting 的痛點，自然演化到 forward model 的 marginalization，利用 temporal redundancy 來邊緣化 nuisance（如 noise、speckle）。目標是實現一個 coding 專案，整合 FAST 的開源代碼和提供的 `xyt_dataset_generator.py`，來驗證 spiral iSCAT 粒子 free diffusion 的 3D 軌跡估計。

**使用建議**：
- 在 Cursor 中，新建一個專案目錄 (e.g., `iscat_fast_mvp`)，將這個 Spec 複製成 `README.md` 或 `spec.md`。
- 使用 Cursor 的 AI 功能 (e.g., "Apply" 或 "Chat")，餵入 Spec，讓它生成/修改 code。
- 逐步執行下面的步驟，Cursor 可以幫你 auto-complete code、debug，或生成 boilerplate。

## 1. 專案背景與來龍去脈 (Context)
### 1.1 原始問題與痛點
- **問題**：在 spiral iSCAT（高速干涉散射顯微）中，追蹤單顆或少數粒子的 free diffusion 軌跡，需要從 noisy 影像序列中估計每幀的 (x(t), y(t), z(t))，尤其是 z(t) 最難（受 SNR 不穩、speckle、背景漂移影響）。
- **現有解法**：逐幀 parametric fitting（基於 analytic interference pattern 的 nonlinear fitting），本質是壓縮的 forward model，但忽略 temporal continuity，易受 noise/mismatch 影響。
- **痛點**：
  - 逐幀對抗 noise，浪費時間結構（粒子軌跡是連續的）。
  - Forward model mismatch 直接毒害結果，無退路。

### 1.2 FAST 啟發與新世界觀
- **FAST 啟發**：FAST (FrAme-multiplexed SpatioTemporal learning strategy) 不精準建模 forward，而是用 sampling + consistency marginalize nuisance。核心：signal 在時空維度可預測，noise 不可預測。
- **類比到 iSCAT**：z(t) 是低頻連續軌跡 (signal)，noise/speckle 是 nuisance → 用 temporal redundancy 逼出 z(t)，而不逐幀 fit。
- **新方案**：用 FAST-like temporal learning + latent trajectory 表示，marginalize nuisance。重點：weak forward constraint (physics gauge)，加上 MSD consistency 作為 identity 破壞測試。
- **MVP 目標**：證明在 imperfect forward model 下，latent + MSD consistency 能 recover 正確 identity 與 z(t) 結構。無 supervision，只用 self-supervised loss。

### 1.3 專案目標與範圍
- **MVP 定義**：用模擬數據 (從 xyt generator 生成) 訓練 FAST-like encoder，輸出 latent z_i(t)，用 MSD loss 確保 consistency。驗證終點：
  - 單粒子：latent smooth, MSD 穩定, z(t) phase-consistent。
  - 多粒子：identity 不 swap, 每條有穩定 MSD signature。
  - Stress test：改 noise/mismatch, MSD 仍穩定。
- **不包含**：重建影像、pixel-wise loss、真實數據橋接（留給 Phase 3）。
- **輸出**：一個 PyTorch 專案，能生成數據、訓練模型、視覺化結果 (e.g., MSD 曲線、軌跡 plot)。

## 2. 技術堆疊與依賴
- **語言/框架**：Python 3.9+, PyTorch 2.0+ (與 FAST 相容)。
- **關鍵組件**：
  - FAST repo: https://github.com/FDU-donglab/FAST (lightweight U-Net + self-supervised loss)。
  - xyt_dataset_generator.py: 提供的 generator code，用來產生 iSCAT-like 影片數據。
- **依賴** (從 FAST 的 requirements.txt + 你的 generator):
  ```
  numpy==1.24.1
  torch==2.5.1
  torchvision==0.20.1
  torchaudio==2.5.1
  scikit-image==0.24.0
  tqdm==4.66.5
  pyqt5==5.15.7  # 如果用 GUI，可選
  csbdeep==0.8.1
  matplotlib  # 加這個用來 plot MSD/軌跡
  tifffile  # generator 需要
  ```
- **環境設置**：用 Conda (如 FAST 建議)。

## 3. 專案結構建議
在 Cursor 中新建目錄結構：
```
iscat_fast_mvp/
├── README.md  # 這個 Spec
├── main.py  # 入口：生成數據、訓練、測試
├── train.py  # 訓練邏輯 (從 FAST 修改)
├── test.py  # 測試/視覺化 (從 FAST 修改)
├── models/  # 從 FAST clone: Unet_Lite.py, loss/loss.py
├── datasets/  # 從 FAST clone + 整合 generator
│   └── xyt_dataset_generator.py  # 你的 generator
├── utils/  # 從 FAST clone
├── data/  # 生成的數據: train/ (tif stacks), test/
├── checkpoint/  # 模型權重
├── result/  # 輸出: latent 軌跡, MSD plots
├── params.json  # 配置 (從 FAST 修改，加 iSCAT params)
└── requirements.txt  # 上述依賴
```

## 4. 實施步驟 (Step-by-Step Guide)
用 Cursor 的 AI 幫你生成 code，按步驟執行。

### 步驟 1: Clone FAST 並設置環境
- 在終端 (或 Cursor 內建終端)：
  ```
  git clone https://github.com/FDU-donglab/FAST.git
  cd FAST
  conda create -n iscat_mvp python=3.9 -y
  conda activate iscat_mvp
  pip install -r requirements.txt
  ```
- 複製 FAST 的 models/, utils/, datasets/ 到你的專案。
- 加你的 `xyt_dataset_generator.py` 到 datasets/。
- 修改 params.json：加 iSCAT params 如 'n_particles': 1, 'z_range': [70e-9, 70e-9], 'photon_scale_range': [40000, 40000]。

### 步驟 2: 生成數據 (用 xyt generator)
- 在 main.py 中呼叫 generator 生成 tif stacks。
- 示例 code (讓 Cursor 生成完整版)：
  ```python
  from datasets.xyt_dataset_generator import EnhancedSimulationPipeline

  def generate_data(config):
      optical_params = {  # 從你的對話抄 iSCAT params
          'wavelength': 532e-9, 'NA': 1.4, 'scattering_model': 'rayleigh',
          # ... 其他 params
      }
      pipeline = EnhancedSimulationPipeline(optical_params, noise_params={})
      rough_surfaces, _ = pipeline.generate_rough_surfaces(n_surfaces=10)
      particle_coords_list = pipeline.generate_particle_coordinates(n_configs=10, n_particles=1)  # Phase 1: 單粒子
      dataset_2d = pipeline.generate_integrated_dataset_with_variations(particle_coords_list, rough_surfaces)
      dataset_xyt = pipeline.generate_xyt_dataset(dataset_2d, n_frames=100)  # 100 幀影片
      # 保存到 data/train/ 作為 tif
      return dataset_xyt
  ```
- 輸出：noisy_movies [N, T, H, W] 作為輸入，coords_physical [N, particles, T, 3] 作為 latent GT (但無 supervision，只用計算經驗 MSD)。

### 步驟 3: 修改 FAST 模型為你的 Encoder
- 用 FAST 的 Unet_Lite 作為 backbone，輸入 frame stack {I_{t-k} ... I_{t+k}}。
- 加 latent head：輸出 z_i(t) (e.g., 低維向量)。
- 示例 (在 models/Unet_Lite.py 修改)：
  ```python
  import torch.nn as nn

  class ISCATEncoder(nn.Module):  # 繼承 FAST 的 Unet_Lite
      def __init__(self):
          super().__init__()
          self.backbone = Unet_Lite()  # 從 FAST
          self.trajectory_head = nn.Linear(hidden_dim, latent_dim)  # 输出 z_i(t)

      def forward(self, frame_stack):  # [B, window_size, H, W]
          features = self.backbone(frame_stack)
          latent = self.trajectory_head(features)  # [B, T, latent_dim]
          return latent
  ```

### 步驟 4: 實現 MSD Consistency Loss (方案 B)
- 在 loss/loss.py 加你的 loss。
- 示例：
  ```python
  def msd_consistency_loss(latent_z, empirical_msd_func):
      h_i = torch.mean(latent_z, dim=1)  # Pooling to embedding [B, embed_dim]
      predicted_msd = neural_g(h_i, tau_range)  # neural_g: MLP(h_i, τ) → MSD(τ)
      empirical_msd = empirical_msd_func(latent_z)  # 從 latent_z 計算 Δz(τ)^2 的經驗平均
      return torch.mean((empirical_msd - predicted_msd)**2)
  ```
- 整合 FAST 的 consistency loss：總 loss = FAST_loss + λ * MSD_loss。

### 步驟 5: 訓練與測試 Pipeline
- 在 train.py 修改：載入數據，用 encoder 輸出 latent，計算 loss (self-supervised + MSD)。
- 加 constraints：smoothness (e.g., TV loss on z(t)), oscillatory (e.g., Fourier domain loss)。
- 测试：生成 MSD 曲線，檢查 swap loss 爆炸。
- 示例 main.py：
  ```python
  if __name__ == "__main__":
      config = load_config('params.json')
      data = generate_data(config)
      model = ISCATEncoder()
      optimizer = torch.optim.Adam(model.parameters())
      for epoch in range(config['epochs']):
          latent = model(data['noisy_movies'])
          loss = fast_loss + msd_consistency_loss(latent, compute_empirical_msd)
          optimizer.step()
      visualize_msd(latent)  # Plot for paper
  ```

### 步驟 6: 驗證 MVP 終點
- 寫 test.py：計算 smoothness (e.g., diff(z(t))), MSD 穩定 (plot 曲線), identity test (swap 軌跡, check loss ↑)。
- Stress test：改 generator params (noise, mismatch), re-train, compare。

## 5. 潛在挑戰與 Debug Tips
- **相容性**：FAST 用 3D data (xy-t)，你的 generator 輸出 XYT，完美 match。
- **Debug**：用 matplotlib plot 軌跡/MSD；在 Cursor 用 "Debug with AI"。
- **擴展**：Phase 2 加多粒子 (n_particles>1)；Phase 3 用真實數據替換 generator。

## Running on Google Colab (Direct Upload)

To run this project on Google Colab without linking Google Drive:

1.  **Zip the Project**: Zip the entire `iscat_fast_mvp` folder. Name it `iscat_fast_mvp.zip`.
2.  **Open Notebook**: Open `colab_runner.ipynb` in Colab.
3.  **Upload Zip**: In the Colab left sidebar ("Files" icon), click the Upload button and select your `iscat_fast_mvp.zip`.
4.  **Run**: Execute the notebook cells to unzip and start training.

> **Note**: Data/Results will be lost when the runtime disconnects. Download `result/` folder if you want to keep them.


