import torch
import torch.nn as nn
from .Unet_Lite import Unet_Lite

class MSDPredictor(nn.Module):
    def __init__(self, input_dim, num_taus=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_taus), # Predicts MSD value for each tau
            nn.Softplus() # Enforce non-negative output for physical consistency
        )

    def forward(self, x):
        return self.net(x)

class ISCATEncoder(Unet_Lite):
    """
    Wraps Unet_Lite to add a latent trajectory head for iSCAT z-position estimation.
    Returns: reconstruction, latent_z, predicted_msd
    """
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='cbr', num_groups=8, latent_dim=1, num_taus=5, **kwargs):
        # Fix for GroupNorm: ensure in_channels is divisible by num_groups (8)
        # We will pad the input from in_channels (5) to 8 internally if needed
        self.original_in_channels = in_channels
        self.padded_in_channels = 8 if in_channels < 8 else (in_channels + 7) // 8 * 8
        
        # Initialize parent Unet_Lite with padded channels
        super().__init__(self.padded_in_channels, out_channels, final_sigmoid, f_maps, layer_order, num_groups, **kwargs)
        
        self.latent_dim = latent_dim
        # Bottleneck feature dimension is the last element of f_maps
        bottleneck_dim = self.f_maps[-1]
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1. Trajectory Head: Features -> z(t)
        self.trajectory_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 2. MSD Predictor Head: Features -> Predicted MSD(tau)
        self.msd_predictor = MSDPredictor(bottleneck_dim, num_taus=num_taus)

        # 3. Bayesian Uncertainty Parameters (Unifying loss from first principles)
        # Represents log(sigma^2). Initializing at 0 means sigma=1.
        self.log_var_fast = nn.Parameter(torch.zeros(1))
        self.log_var_msd = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        time_num = x.shape[1]
        
        # Determine padding needed
        if x.shape[1] < self.padded_in_channels:
            padding_channels = self.padded_in_channels - x.shape[1]
            # Pad in the channel dimension (dim=1)
            # F.pad format: (left, right, top, bottom, front, back) for 3D, but here x is 4D (B, C, H, W)
            # We want to pad dim 1.
            # Construct zero padding
            zero_pad = torch.zeros(x.shape[0], padding_channels, x.shape[2], x.shape[3], device=x.device)
            x_padded = torch.cat([x, zero_pad], dim=1)
        else:
            x_padded = x
            
        # --- Encoder Path (Copying logic from Unet_Lite.forward) ---
        encoders_features = []
        curr_x = x_padded
        for encoder in self.encoders:
            curr_x = encoder(curr_x)
            encoders_features.insert(0, curr_x)
            
        # x is now the output of the last encoder (bottleneck features)
        bottleneck_features = curr_x
        
        # --- Heads ---
        # Global Average Pooling on the bottleneck
        pooled = self.avg_pool(bottleneck_features).view(bottleneck_features.size(0), -1)
        
        # Latent z(t)
        latent = self.trajectory_head(pooled)
        
        # Predicted MSD Parameters (for this time point)
        pred_msd = self.msd_predictor(pooled)
        
        # --- Decoder Path ---
        encoders_features = encoders_features[1:] # Drop the first
        
        decoder_x = bottleneck_features
        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):
            use_skip_connections = i != 0
            decoder_x = decoder(encoder_features, decoder_x, use_skip_connections)

        decoder_x = self.final_conv(decoder_x)
        
        # Return all
        return decoder_x, latent, pred_msd
        