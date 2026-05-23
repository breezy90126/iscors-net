import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt

# Import our newly created modules
from datasets.tau_sparse_dataset import TauSparseDataset
from models.pissl_tau_encoder import PISSLTauEncoder
from loss.physics_loss import PhysicsInformedLoss

def train_internal_learning():
    # ==========================================
    # 1. Configuration
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 30
    batch_size = 4
    learning_rate = 1e-4
    tau_delays = (0, 1, 2, 4, 8, 16, 32, 64)
    patch_size = 64
    
    checkpoint_dir = './checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Loading 'Real' Cell Video (Generated test video)...")
    import tifffile
    video_path = "./data/test_synthetic_cell.tif"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Please run utils/generate_test_video.py first.")
        return
        
    video_matrix = tifffile.imread(video_path).astype(np.float32)
    T, H, W = video_matrix.shape
    print(f"Loaded video with shape: (T={T}, H={H}, W={W})")
    
    # We create the dataset. This dataset will automatically sample 1% of the pixels
    # and provide dummy GT for them.
    train_dataset = TauSparseDataset(
        video_tensor=video_matrix,
        tau_delays=tau_delays,
        patch_size=patch_size,
        mode='train',
        train_ratio=0.10  # 10% internal learning for higher quality
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # ==========================================
    # 3. Model & Loss Setup
    # ==========================================
    model = PISSLTauEncoder(num_tau_channels=len(tau_delays)).to(device)
    criterion = PhysicsInformedLoss(lambda_gamma=1.0, lambda_alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ==========================================
    # 4. Training Loop (Internal Learning)
    # ==========================================
    print(f"Starting Internal Learning on 10% sparse data ({len(train_dataset)} patches/epoch)...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_gamma_loss = 0.0
        epoch_alpha_loss = 0.0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            # inputs: (B, 8, P, P)
            # targets: (B, 2, P, P) - Channel 0 is Gamma, 1 is Alpha
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(inputs)
            
            # Calculate Physics Loss
            loss, loss_gamma, loss_alpha = criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_gamma_loss += loss_gamma.item()
            epoch_alpha_loss += loss_alpha.item()
            
            pbar.set_postfix({'Loss': loss.item(), 'G': loss_gamma.item(), 'A': loss_alpha.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # Save the internally trained model
    save_path = os.path.join(checkpoint_dir, 'pissl_internal_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    
    # ==========================================
    # 5. Full Frame Inference Demonstration
    # ==========================================
    print("\n--- Starting Full Frame Inference ---")
    model.eval()
    
    # We create an inference dataset (fetches the full frame instead of patches)
    inference_dataset = TauSparseDataset(
        video_tensor=video_matrix,
        tau_delays=tau_delays,
        mode='inference'
    )
    
    # The dataset returns a single tensor of shape (8, H, W)
    full_frame_input = inference_dataset[0].unsqueeze(0).to(device) # Add batch dim: (1, 8, H, W)
    
    with torch.no_grad():
        preds_full = model(full_frame_input) # Output: (1, 2, H, W)
        
    gamma_map = preds_full[0, 0, :, :].cpu().numpy()
    alpha_map = preds_full[0, 1, :, :].cpu().numpy()
    
    print(f"Inference complete!")
    print(f"Gamma Map shape: {gamma_map.shape}, mean: {gamma_map.mean():.3f}")
    print(f"Alpha Map shape: {alpha_map.shape}, mean: {alpha_map.mean():.3f}")
    
    # Optional: Save dummy visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gamma_map, cmap='magma')
    plt.title('Predicted Gamma Map')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(alpha_map, cmap='viridis')
    plt.title('Predicted Alpha Map')
    plt.colorbar()
    
    os.makedirs('./result', exist_ok=True)
    plt.savefig('./result/inference_maps.png')
    print("Saved inference visualization to ./result/inference_maps.png")

if __name__ == "__main__":
    train_internal_learning()
