import torch
import torch.nn as nn


class PhysicsReconLoss(nn.Module):
    """
    Self-supervised Physics Reconstruction loss.

    Modes (shape_only / legacy_2ch / default):
      shape_only=True  [v3.4+]: target pre-normalised by dataset (G_norm starts at 1).
                                G_theory_norm = (1+γ)/(1+γτ^α).
      legacy_2ch=True  [v3.1]:  both sides normalised inside the loss.
      default          [v3.2/v3.3]: amplitude-aware A/(1+γτ^α).

    Weighting options (applied after mode selection):
      tau_weighted=True [v3.6]: w_k = log(τ_k), normalised to sum=1.
                                Up-weights large-τ terms where α signal is
                                strongest; more stable than log-MSE.
                                τ=1 naturally gets weight 0 (log(1)=0).
      log_space=True   [v3.5]: log-MSE instead of linear MSE.
    """

    def __init__(self, recon_taus, shape_only=False, legacy_2ch=False,
                 log_space=False, tau_weighted=False):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))          # (1,1,1,K)

        # τ-weights: log(τ_k), normalised to sum=1; τ=1 → weight=0
        w = torch.log(taus.clamp(min=1.0))                             # (K,)
        w = w / (w.sum() + 1e-10)
        self.register_buffer("tau_weights", w.view(1, 1, 1, -1))      # (1,1,1,K)

        self.shape_only    = shape_only
        self.legacy_2ch    = legacy_2ch
        self.log_space     = log_space
        self.tau_weighted  = tau_weighted

    def forward(self, preds, g_empirical, train_mask):
        """
        Args:
            preds:       (B, 2, H, W) [γ,α] for shape_only/legacy_2ch;
                         (B, 3, H, W) [γ,α,A] for amplitude mode.
            g_empirical: (B, H, W, K)
            train_mask:  (B, H, W) — 1.0 at supervised pixels.
        """
        gamma = preds[:, 0].unsqueeze(-1)   # (B, H, W, 1)
        alpha = preds[:, 1].unsqueeze(-1)
        eps   = 1e-10

        g_theory = 1.0 / (1.0 + gamma * torch.pow(self.taus, alpha))  # (B,H,W,K)

        if self.shape_only:
            g_theory = g_theory / (g_theory[..., 0:1] + eps)

        elif self.legacy_2ch:
            g_theory    = g_theory    / (g_theory[..., 0:1] + eps)
            g_empirical = g_empirical / (g_empirical[..., 0:1].abs() + eps)

        else:
            amp = preds[:, 2].unsqueeze(-1)
            g_theory = amp / (1.0 + gamma * torch.pow(self.taus, alpha))

        if self.log_space:
            sq_err = (torch.log(g_theory.clamp(min=1e-8))
                      - torch.log(g_empirical.abs().clamp(min=1e-8))) ** 2
        else:
            sq_err = (g_theory - g_empirical) ** 2                     # (B,H,W,K)

        mask4d = train_mask.unsqueeze(-1)                              # (B,H,W,1)

        if self.tau_weighted:
            # Weighted average over τ per pixel, then average over supervised pixels
            # tau_weights sums to 1 (τ=1 gets weight 0)
            return (sq_err * mask4d * self.tau_weights).sum() / (mask4d.sum() + eps)
        else:
            n_terms = mask4d.sum() * self.taus.numel() + eps
            return (sq_err * mask4d).sum() / n_terms


if __name__ == "__main__":
    import torch
    B, H, W, K = 2, 64, 64, 10
    taus = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    mask = torch.ones(B, H, W); mask[:, ::3, ::3] = 0.0

    p2     = torch.cat([torch.rand(B,1,H,W)*0.5, torch.rand(B,1,H,W)*2], dim=1)
    g_norm = torch.rand(B, H, W, K); g_norm[..., 0] = 1.0

    L_unweighted = PhysicsReconLoss(taus, shape_only=True)(p2, g_norm, mask)
    L_weighted   = PhysicsReconLoss(taus, shape_only=True, tau_weighted=True)(p2, g_norm, mask)
    L_log        = PhysicsReconLoss(taus, shape_only=True, log_space=True)(p2, g_norm, mask)

    print(f"shape_only unweighted : {L_unweighted.item():.4e}")
    print(f"shape_only τ-weighted : {L_weighted.item():.4e}")
    print(f"shape_only log-MSE    : {L_log.item():.4e}")
    w = torch.log(torch.tensor(taus, dtype=torch.float32).clamp(min=1))
    w = w / w.sum()
    print(f"τ-weights: {w.numpy().round(3)}")
