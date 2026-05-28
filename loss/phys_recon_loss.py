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
      fisher_weighted=True [v3.8]: w_k ∝ (∂G_norm/∂γ)² + (∂G_norm/∂α)² at prior (γ₀,α₀).
                                   Fisher information measures how much each τ contributes
                                   to identifying BOTH parameters jointly. Peaks at
                                   intermediate τ; naturally zeros τ=1; avoids over-weighting
                                   saturated large-τ where G≈0.
      tau_weighted=True    [v3.6]: w_k = log(τ_k). Legacy option; biases toward large τ
                                   and suppresses small-τ γ gradient.
      log_space=True       [v3.5]: log-MSE instead of linear MSE.
    """

    def __init__(self, recon_taus, shape_only=False, legacy_2ch=False,
                 log_space=False, tau_weighted=False,
                 fisher_weighted=False, fisher_gamma_prior=0.1, fisher_alpha_prior=1.0):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))          # (1,1,1,K)

        # Legacy log(τ) weights
        w_log = torch.log(taus.clamp(min=1.0))
        w_log = w_log / (w_log.sum() + 1e-10)
        self.register_buffer("tau_weights", w_log.view(1, 1, 1, -1))  # (1,1,1,K)

        # Fisher information weights at prior (γ₀, α₀)
        w_fisher = self._fisher_weights(taus, fisher_gamma_prior, fisher_alpha_prior)
        self.register_buffer("fisher_weights", w_fisher.view(1, 1, 1, -1))

        self.shape_only      = shape_only
        self.legacy_2ch      = legacy_2ch
        self.log_space       = log_space
        self.tau_weighted    = tau_weighted
        self.fisher_weighted = fisher_weighted

    @staticmethod
    def _fisher_weights(taus, gamma_0=0.1, alpha_0=1.0):
        """
        τ-weights proportional to Fisher information of G_norm(τ) at prior (γ₀, α₀).

        G_norm(τ) = (1+γ) / (1+γτ^α)   →   normalization at τ=1 always gives G_norm=1.

        Partial derivatives:
            ∂G_norm/∂γ = (1 − τ^α) / (1 + γτ^α)²
            ∂G_norm/∂α = −(1+γ)·γ·τ^α·log(τ) / (1 + γτ^α)²

        Fisher information: f_k = (∂G_norm/∂γ)² + (∂G_norm/∂α)²

        Properties:
          τ=1: both partials = 0 → weight = 0  (normalization point, zero information)
          Small τ: dominated by ∂G/∂γ  (γ-sensitive zone)
          Large τ: dominated by ∂G/∂α  (α-sensitive zone, until G saturates to 0)
          Saturated τ (G≈0): both partials → 0 → weight → 0  (automatic saturation masking)
        """
        t_a   = taus.clamp(min=1e-8) ** alpha_0
        denom = 1.0 + gamma_0 * t_a

        dG_dg = (1.0 - t_a) / denom ** 2
        dG_da = (-(1.0 + gamma_0) * gamma_0 * t_a
                 * torch.log(taus.clamp(min=1.0)) / denom ** 2)

        fisher = dG_dg ** 2 + dG_da ** 2
        fisher = fisher / (fisher.sum() + 1e-10)
        return fisher

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

        if self.fisher_weighted:
            # Fisher-weighted average over τ, then average over supervised pixels
            return (sq_err * mask4d * self.fisher_weights).sum() / (mask4d.sum() + eps)
        elif self.tau_weighted:
            return (sq_err * mask4d * self.tau_weights).sum() / (mask4d.sum() + eps)
        else:
            n_terms = mask4d.sum() * self.taus.numel() + eps
            return (sq_err * mask4d).sum() / n_terms


if __name__ == "__main__":
    import torch
    import numpy as np

    taus = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    taus_t = torch.tensor(taus, dtype=torch.float32)

    # Compare weight profiles
    w_log = torch.log(taus_t.clamp(min=1)); w_log /= w_log.sum()
    w_fisher = PhysicsReconLoss._fisher_weights(taus_t, gamma_0=0.1, alpha_0=1.0)

    print("τ       log(τ)   Fisher(γ₀=0.1, α₀=1.0)")
    for i, tau in enumerate(taus):
        print(f"  {tau:3d}   {w_log[i]:.4f}   {w_fisher[i]:.4f}")

    # Sanity: Fisher weights sum to 1, τ=1 gets 0
    assert abs(w_fisher.sum().item() - 1.0) < 1e-5
    assert w_fisher[0].item() < 1e-8, "τ=1 should have zero Fisher weight"
    print(f"\nFisher sum={w_fisher.sum():.6f}  τ=1 weight={w_fisher[0]:.2e}  ✓")

    # Forward pass check
    B, H, W, K = 2, 64, 64, 10
    mask   = torch.ones(B, H, W); mask[:, ::3, ::3] = 0.0
    preds  = torch.cat([torch.rand(B,1,H,W)*0.5, torch.rand(B,1,H,W)*2], dim=1)
    g_norm = torch.rand(B, H, W, K); g_norm[..., 0] = 1.0

    L_unweighted = PhysicsReconLoss(taus, shape_only=True)(preds, g_norm, mask)
    L_log_tau    = PhysicsReconLoss(taus, shape_only=True, tau_weighted=True)(preds, g_norm, mask)
    L_fisher     = PhysicsReconLoss(taus, shape_only=True, fisher_weighted=True)(preds, g_norm, mask)

    print(f"\nshape_only uniform    : {L_unweighted.item():.4e}")
    print(f"shape_only log(τ)     : {L_log_tau.item():.4e}")
    print(f"shape_only Fisher     : {L_fisher.item():.4e}")
