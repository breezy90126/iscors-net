import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.sequence_dataset import ISCATSequenceDataset
from datasets.data_process import sampler, generate_subimages
from models.iscat_encoder import ISCATEncoder
from models.loss.msd_loss import MSDConsistencyLoss
import json
import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(config):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    # Expects data in config['train_folder']
    train_dataset = ISCATSequenceDataset(
        dataPath=config['train_folder'], 
        mode='train', 
        stack_size=5, # Should be odd usually
        sequence_length=config.get('sequence_length', 20) # Define in config or default
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0, # config['num_workers']
        drop_last=True
    )
    
    # Model
    # input_channels = stack_size
    model = ISCATEncoder(
        in_channels=5, # stack_size
        out_channels=5, # stack_size (reconstruction) or 1? FAST reconstructs same channels? 
        # FAST Unet_Lite: in_channels=miniBatch(frames), out_channels=miniBatch.
        final_sigmoid=False,
        latent_dim=1
    ).to(device)
    
    # Losses
    msd_criterion = MSDConsistencyLoss(tau_range=[1, 2, 3, 4, 5])
    reconstruction_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    epochs = config['epochs']
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (seq_data, gt_z) in enumerate(pbar):
            # seq_data: [B, T_seq, Stack, H, W]
            # gt_z: [B, T_seq, N_particles]
            seq_data = seq_data.to(device)
            B, T_seq, Stack, H, W = seq_data.shape
            
            # 1. Processing for MSD Loss (Full Resolution)
            # Normalize: Raw / Median - 1.0 (Background Correction)
            # "Video Medium" -> Temporal Median over the sequence
            median_bg = torch.median(seq_data, dim=1, keepdim=True).values # [B, 1, Stack, H, W]
            seq_data_norm = seq_data / (median_bg + 1e-6) - 1.0
            
            # Flatten to [B*T_seq, Stack, H, W]
            flat_input = seq_data_norm.view(-1, Stack, H, W)
            # No further normalization needed as we centered it around 0 via (-1.0)
            
            # Forward pass on full data
            recons_full, latent_full, pred_msd_full = model(flat_input)
            
            # latent_full: [B*T_seq, 1]
            latent_seq = latent_full.view(B, T_seq)
            
            # Prediction: [B*T_seq, num_taus] -> Average over sequence for stability?
            # Or enforce consistency at every time point? 
            # Spec says: "predicted_msd = neural_g(h_i, tau)"
            # Let's average predictions over the sequence for the MSD loss
            pred_msd_seq = pred_msd_full.view(B, T_seq, -1)
            pred_msd_mean = torch.mean(pred_msd_seq, dim=1) # [B, num_taus]
            
            # MSD Loss
            # Consistency: Empirical MSD of `latent_seq` should match `pred_msd_mean`
            loss_msd_raw = msd_criterion(latent_seq.unsqueeze(-1), pred_msd_mean)
            
            # 2. Processing for FAST Reconstruction Loss (Subsampled)
            # Generate masks
            # seed for reproducibility in iteration
            curr_seed = batch_idx
            mask1, mask2 = sampler(flat_input, operation_seed_counter=curr_seed)
            input_sub1 = generate_subimages(flat_input, mask1) # [N, Stack, H/2, W/2]
            target_sub2 = generate_subimages(flat_input, mask2)
            
            # Forward on sub1
            recons_sub1, _, _ = model(input_sub1)
            
            # FAST Loss
            loss_fast_raw = reconstruction_criterion(recons_sub1, target_sub2)
            
            # 3. Bayesian Unified Loss (Kendall et al.)
            # loss = (1/(2*sigma^2)) * raw_loss + log(sigma)
            # We use log_var = log(sigma^2) for stability
            
            precision_fast = torch.exp(-model.log_var_fast)
            precision_msd = torch.exp(-model.log_var_msd)
            
            loss_fast = precision_fast * loss_fast_raw + model.log_var_fast
            loss_msd = precision_msd * loss_msd_raw + model.log_var_msd
            
            # Total Loss
            loss = loss_fast + loss_msd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Fast': loss_fast_raw.item(), 'MSD': loss_msd_raw.item(), 's_f': torch.exp(model.log_var_fast).item(), 's_m': torch.exp(model.log_var_msd).item()})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss/len(train_loader)}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            save_path = os.path.join(config['checkpoint_path'], f"checkpoint_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    if os.path.exists('params.json'):
        config = load_config('params.json')
        train(config)
    else:
        print("params.json not found")
