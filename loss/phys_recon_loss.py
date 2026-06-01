import torch
import torch.nn as nn


class PhysicsReconLoss(nn.Module):
    """
    Self-supervised Physics Reconstruction loss.

    Modes (shape_only / legacy_2ch / default):
      shape_only=True  [v3.4+]: target pre-normalised by dataset.
                                G_theory_norm = (1+γ·τ_ref^α)/(1+γτ^α)  where
                                τ_ref = recon_taus[0] (first tau = normalization anchor).
                                For recon_taus[0]=1: reduces to (1+γ)/(1+γτ^α) (legacy).
      legacy_2ch=True  [v3.1]:  both sides normalised inside the loss.
      default          [v3.2/v3.3]: amplitude-aware A/(1+γτ^α).

    τ selection and normalization anchor:
      recon_taus[0] is the normalization reference τ_ref.
      Dataset:   G_norm(τ) = G_empirical(τ) / G_empirical(τ_ref)
      Loss:      G_theory_norm(τ) = G_theory(τ) / G_theory(τ_ref) = (1+γτ_ref^α)/(1+γτ^α)
      Fisher:    τ_ref gets weight=0 (normalization point; zero identifiability).

      To exclude noisy small-τ channels (e.g., τ=1,2,4,8 for fast real dynamics):
        recon_taus = (16, 32, 48, 64, 96, 128)   → τ_ref=16, K=6
      No other code changes needed — dataset, model (τ-PE), and loss auto-adapt.

    Weighting options (applied after mode selection):
      fisher_weighted=True [v3.8]: w_k ∝ (∂G_norm/∂γ)² + (∂G_norm/∂α)² at prior (γ₀,α₀).
                                   General formula for any τ_ref. Peaks at intermediate τ
                                   where both parameters are identifiable; zeros at τ_ref
                                   and at saturated large-τ where G≈0.
      tau_weighted=True    [v3.6]: w_k = log(τ_k). Legacy; biases toward large τ.
      log_space=True       [v3.5]: log-MSE instead of linear MSE.

    Reliability weighting [v4.1]:
      forward() accepts an optional sigma_g_norm argument: (B, H, W, K) tensor of
      per-pixel normalised measurement noise σ_G_norm(τ;y,x).

      Reliability weight: r(τ;y,x) = 1 / (σ_G_norm(τ;y,x) + ε)
      Combined weight  : w_combined(τ;y,x) = fisher(τ) × r(τ;y,x)
      Normalised per pixel so Σ_τ w_combined = 1 at each (y,x).

      When sigma_g_norm is None (default), falls back to Fisher-only or uniform
      weighting as before, preserving full backward compatibility.
    """

    def __init__(self, recon_taus, shape_only=False, legacy_2ch=False,
                 log_space=False, tau_weighted=False,
                 fisher_weighted=False, fisher_gamma_prior=0.1, fisher_alpha_prior=1.0):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))          # (1,1,1,K)

        # τ_ref = first tau = G_norm normalization anchor (same as dataset)
        tau_ref = float(recon_taus[0])

        # Legacy log(τ) weights
        w_log = torch.log(taus.clamp(min=1.0))
        w_log = w_log / (w_log.sum() + 1e-10)
        self.register_buffer("tau_weights", w_log.view(1, 1, 1, -1))  # (1,1,1,K)

        # Fisher information weights at prior (γ₀, α₀) with correct τ_ref
        w_fisher = self._fisher_weights(taus, fisher_gamma_prior, fisher_alpha_prior,
                                        tau_ref=tau_ref)
        self.register_buffer("fisher_weights", w_fisher.view(1, 1, 1, -1))

        self.shape_only      = shape_only
        self.legacy_2ch      = legacy_2ch
        self.log_space       = log_space
        self.tau_weighted    = tau_weighted
        self.fisher_weighted = fisher_weighted

    @staticmethod
    def _fisher_weights(taus, gamma_0=0.1, alpha_0=1.0, tau_ref=1.0):
        """
        τ-weights ∝ Fisher information of G_norm(τ; τ_ref) at prior (γ₀, α₀).

        G_norm(τ; τ_ref) = (1 + γ·τ_ref^α) / (1 + γ·τ^α)

        Partial derivatives (general τ_ref):
            ∂G_norm/∂γ = (τ_ref^α − τ^α) / (1 + γτ^α)²
            ∂G_norm/∂α = γ·[τ_ref^α·log(τ_ref)·(1+γτ^α)
                            − (1+γτ_ref^α)·τ^α·log(τ)] / (1 + γτ^α)²

        Fisher information: f_k = (∂G_norm/∂γ)² + (∂G_norm/∂α)²

        Properties (valid for any τ_ref):
          τ=τ_ref: both partials = 0 → weight = 0  (normalization point, no information)
          Small τ (> τ_ref): dominated by ∂G/∂γ  (γ-sensitive zone)
          Large τ: dominated by ∂G/∂α  (α-sensitive zone)
          Saturated τ (G≈0): both partials → 0 → weight → 0

        Special case τ_ref=1: reduces to legacy formula
            ∂G_norm/∂γ = (1 − τ^α) / (1 + γτ^α)²
            ∂G_norm/∂α = −(1+γ)·γ·τ^α·log(τ) / (1 + γτ^α)²
        """
        import math
        tau_ref_a   = tau_ref ** alpha_0                           # scalar
        t_a         = taus.clamp(min=1e-8) ** alpha_0             # (K,)
        D           = 1.0 + gamma_0 * t_a                         # 1 + γ·τ^α
        D2          = D ** 2

        dG_dg = (tau_ref_a - t_a) / D2

        log_tau_ref = math.log(max(tau_ref, 1.0))                 # 0 when τ_ref=1
        log_taus    = torch.log(taus.clamp(min=1.0))
        N_ref       = 1.0 + gamma_0 * tau_ref_a                   # 1 + γ·τ_ref^α
        dG_da = (gamma_0 * tau_ref_a * log_tau_ref * D
                 - N_ref * gamma_0 * t_a * log_taus) / D2

        fisher = dG_dg ** 2 + dG_da ** 2
        fisher = fisher / (fisher.sum() + 1e-10)
        return fisher

    def forward(self, preds, g_empirical, train_mask, sigma_g_norm=None):
        """
        Args:
            preds:          (B, 2, H, W) [γ,α] for shape_only/legacy_2ch;
                            (B, 3, H, W) [γ,α,A] for amplitude mode.
            g_empirical:    (B, H, W, K)
            train_mask:     (B, H, W) — 1.0 at supervised pixels.
            sigma_g_norm:   (B, H, W, K) optional — per-pixel normalised σ_G.
                            When provided (v4.1), combined Fisher × Reliability
                            weights are used (fisher_weighted must be True).
                            When None, falls back to original fisher/tau/uniform.
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

        if self.fisher_weighted and sigma_g_norm is not None:
            # Combined Fisher × Reliability weighting (v4.1)
            # reliability(τ;y,x) = 1 / (σ_G_norm + ε) — inverse noise
            reliability = 1.0 / (sigma_g_norm + eps)                   # (B,H,W,K)
            # Combine: Fisher (1,1,1,K) × reliability (B,H,W,K)
            combined = self.fisher_weights * reliability                # (B,H,W,K)
            # Normalise per pixel so weights sum to 1 over τ
            combined = combined / (combined.sum(dim=-1, keepdim=True) + eps)
            return (sq_err * mask4d * combined).sum() / (mask4d.sum() + eps)

        elif self.fisher_weighted:
            return (sq_err * mask4d * self.fisher_weights).sum() / (mask4d.sum() + eps)
        elif self.tau_weighted:
            return (sq_err * mask4d * self.tau_weights).sum() / (mask4d.sum() + eps)
        else:
            n_terms = mask4d.sum() * self.taus.numel() + eps
            return (sq_err * mask4d).sum() / n_terms


if __name__ == "__main__":
    import torch
    import numpy as np

    # ── Case 1: legacy τ_ref=1 (recon_taus starts at 1) ─────────────────────
    taus_full = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    taus_t = torch.tensor(taus_full, dtype=torch.float32)

    w_log    = torch.log(taus_t.clamp(min=1)); w_log /= w_log.sum()
    w_fisher = PhysicsReconLoss._fisher_weights(taus_t, gamma_0=0.1, alpha_0=1.0, tau_ref=1.0)

    print("=== τ_ref=1  (full τ set: 1..128) ===")
    print(f"{'τ':>5}  {'log(τ)':>8}  {'Fisher':>8}")
    for i, tau in enumerate(taus_full):
        print(f"  {tau:3d}   {w_log[i]:.4f}   {w_fisher[i]:.4f}")

    assert abs(w_fisher.sum().item() - 1.0) < 1e-5
    assert w_fisher[0].item() < 1e-8, "τ=τ_ref should have zero Fisher weight"
    print(f"  sum={w_fisher.sum():.6f}  τ=1 weight={w_fisher[0]:.2e}  ✓")

    # ── Case 2: τ_ref=16 (drop noisy small-τ for fast real dynamics) ─────────
    taus_fast = [16, 32, 48, 64, 96, 128]
    taus_f = torch.tensor(taus_fast, dtype=torch.float32)
    w_fisher_16 = PhysicsReconLoss._fisher_weights(taus_f, gamma_0=0.1, alpha_0=1.0, tau_ref=16.0)

    print("\n=== τ_ref=16  (fast-dynamics τ set: 16..128) ===")
    print(f"{'τ':>5}  {'Fisher(τ_ref=16)':>18}")
    for i, tau in enumerate(taus_fast):
        print(f"  {tau:3d}   {w_fisher_16[i]:.4f}")
    assert abs(w_fisher_16.sum().item() - 1.0) < 1e-5
    assert w_fisher_16[0].item() < 1e-8, "τ=τ_ref=16 should have zero Fisher weight"
    print(f"  sum={w_fisher_16.sum():.6f}  τ=16 weight={w_fisher_16[0]:.2e}  ✓")

    # ── Forward pass check (both τ sets) ─────────────────────────────────────
    for taus_cfg, label in [(taus_full, "K=10 τ_ref=1"), (taus_fast, "K=6 τ_ref=16")]:
        K   = len(taus_cfg)
        B, H, W = 2, 64, 64
        mask   = torch.ones(B, H, W); mask[:, ::3, ::3] = 0.0
        preds  = torch.cat([torch.rand(B,1,H,W)*0.5, torch.rand(B,1,H,W)*2], dim=1)
        g_norm = torch.rand(B, H, W, K); g_norm[..., 0] = 1.0  # τ_ref channel = 1
        L = PhysicsReconLoss(taus_cfg, shape_only=True,
                              fisher_weighted=True)(preds, g_norm, mask)
        print(f"\n[{label}] Fisher loss = {L.item():.4e}  ✓")
