import torch
import torch.nn as nn

def compute_empirical_msd(latent_z, tau_range):
    """
    Computes empirical MSD from latent trajectory z(t).
    latent_z: [B, T, 1] or [B, T]
    tau_range: list of lag times (integers)
    Returns: msd_values [B, len(tau_range)]
    """
    if latent_z.dim() == 3:
        latent_z = latent_z.squeeze(-1) # [B, T]
    
    B, T = latent_z.shape
    msd_list = []
    
    for tau in tau_range:
        if tau >= T:
             # Handle large tau? Just append 0 or nan? Or skip.
             # Ideally tau should be < T
             msd_list.append(torch.zeros(B, device=latent_z.device))
             continue
        diff = latent_z[:, tau:] - latent_z[:, :-tau]
        sq_diff = diff ** 2
        msd = torch.mean(sq_diff, dim=1) # Average over time T-tau
        msd_list.append(msd)
        
    return torch.stack(msd_list, dim=1) # [B, num_taus]

class MSDConsistencyLoss(nn.Module):
    def __init__(self, tau_range=[1, 2, 3, 4, 5]):
        super().__init__()
        self.tau_range = tau_range
        # For now, we compare empirical MSD to... self-consistency?
        # Spec says: "predicted_msd = neural_g(h_i, tau_range)"
        # "return mean((empirical_msd - predicted_msd)**2)"
        # We need a neural_g (the MSD predictor head).
        # This head should be trained.
        
        # But wait, neural_g should be part of the model or loss?
        # Usually part of the model, but here loss needs access to it.
        # Let's assume prediction is done outside and passed in, or we build a small MLP here?
        # "neural_g(h_i, tau)" where h_i is embedding.
        # Ideally, we pass "predicted_msd" and "latent_z" to forward.
        pass

    def forward(self, latent_z, predicted_msd):
        """
        latent_z: [B, T, 1]
        predicted_msd: [B, num_taus]
        """
        empirical_msd = compute_empirical_msd(latent_z, self.tau_range)
        # Normalize/Scale?
        loss = torch.mean((empirical_msd - predicted_msd) ** 2)
        return loss

# We also need a way to integrate this into the main training Loop.
# The ISCATEncoder currently outputs `latent` which is `z(t)` for the chunk?
# If `latent` is [B, 1], we cannot compute MSD from it unless we have a sequence of outputs.
# FAST training loop usually feeds independent batches.
# To compute MSD, we need a sequence of frames.
# Solution: Input to model should be a long sequence, OR we simply train on batches of SHORT sequences (T=100) and model outputs T=100 latent.
# `ISCATEncoder` logic I implemented inputs `[B, C, H, W]` and outputs `[B, 1]` via GAP.
# This assumes `C` frames -> 1 z-value (middle).
# To get `z(t)` for t=1..T, we need to run the model T times (sliding window).
# This is slow for training.
# Better approach for MVP:
# The model takes `[B, T, H, W]` and treats T as time, outputs `[B, T, 1]`.
# Unet_Lite uses 2D convs. `time_num = x.shape[1]`.
# If we treat T as `channels`, they are mixed.
# We probably need 3D Conv or shared weights 2D conv over time.
# FAST `Unet_Lite` mixes channels.
# 
# Re-reading Spec: "Input frame stack {I_{t-k} ... I_{t+k}}. Output z_i(t)."
# This means ONE z value per stack.
# So to train MSD, we need to generate a SEQUENCE of z values: z(t), z(t+1)...
# So we need to run a BATCH of overlapping stacks.
# e.g. Batch item 1: stack centered at t.
# Batch item 2: stack centered at t+1.
# ...
# Then we stack these outputs to get `z(t)...z(t+T)`.
# Then compute MSD on this vector.
# Then Optimization.
#
# This setup requires `dataset` to provide SEQUENTIAL batches or `train.py` to organize them.
# `ReadDatasets` in `dataset.py` currently returns random indices (shuffle).
# For MSD loss, we need TEMPORAL CONTINUITY in the batch.
# We need to modify `dataset.py` or `train.py` to sample "TrajSnippets".
#
# Plan:
# Modify `train.py` to reshuffle or sample custom batches.
# Or simpler: Update `dataset.py` to return a `sequence` of stacks.
# `__getitem__` returns `[T_seq, Window, H, W]`.
# Model forward runs on `[B*T_seq, Window, H, W]` -> `[B*T_seq, 1]`.
# Reshape to `[B, T_seq, 1]` -> Compute MSD.
