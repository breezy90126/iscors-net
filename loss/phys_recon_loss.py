import torch
import torch.nn as nn


class PhysicsReconLoss(nn.Module):
    """
    Self-supervised Physics Reconstruction loss for iSCORS internal learning.

    v3.2 (amplitude-aware): model predicts (γ, α, A) per pixel and
        G_theory(τ) = A / (1 + γ · τ^α)
    is compared directly to the precomputed G_empirical(τ) — no normalisation.

    Why A as a 3rd output instead of normalising:
      • v3.0 had no normalisation → G_theory ~1 vs G_empirical ~1e-3 → saturated.
      • v3.1 normalised at τ=recon_taus[0]; but the anchor value 1/(1+γ) carries
        γ information, so normalisation discards the signal that distinguishes
        cell body (γ≈0.1) from background (γ=0). Cell-body γ collapsed to 0.
      • Predicting A explicitly restores amplitude — A→0 marks background,
        A>0 marks cell — and (γ,α) recover their pure shape role.

    log_space=True: loss = MSE(log G_theory, log G_empirical), equivalent to
    relative MSE. Up-weights small-G (large τ) entries where α's gradient is
    strongest — addresses the ∂G/∂α ∝ γ·log(τ) asymmetry when γ is small.

    legacy_2ch=True: back-compat path that uses v3.1's normalised 2-channel
    loss. For ablation only.
    """

    def __init__(self, recon_taus, log_space=False, legacy_2ch=False):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))
        self.log_space  = log_space
        self.legacy_2ch = legacy_2ch

    def forward(self, preds, g_empirical, train_mask):
        """
        Args:
            preds:       (B, 3, H, W) for v3.2 [γ,α,A]; (B, 2, H, W) in legacy mode.
            g_empirical: (B, H, W, K)
            train_mask:  (B, H, W) — 1.0 at supervised pixels, 0.0 at held-out.
        """
        gamma = preds[:, 0].unsqueeze(-1)            # (B, H, W, 1)
        alpha = preds[:, 1].unsqueeze(-1)

        if self.legacy_2ch:
            g_theory = 1.0 / (1.0 + gamma * torch.pow(self.taus, alpha))
            eps = 1e-10
            g_theory    = g_theory    / (g_theory[..., 0:1] + eps)
            g_empirical = g_empirical / (g_empirical[..., 0:1].abs() + eps)
        else:
            amp = preds[:, 2].unsqueeze(-1)
            g_theory = amp / (1.0 + gamma * torch.pow(self.taus, alpha))

        if self.log_space:
            eps = 1e-8
            sq_err = (torch.log(g_theory.clamp(min=eps))
                      - torch.log(g_empirical.abs().clamp(min=eps))) ** 2
        else:
            sq_err = (g_theory - g_empirical) ** 2

        mask4d = train_mask.unsqueeze(-1)
        n_terms = mask4d.sum() * self.taus.numel() + 1e-10
        loss = (sq_err * mask4d).sum() / n_terms
        return loss


if __name__ == "__main__":
    B, H, W, K = 2, 64, 64, 8
    preds = torch.cat([
        torch.rand(B, 1, H, W),          # γ
        torch.rand(B, 1, H, W) * 2.0,    # α
        torch.full((B, 1, H, W), 1e-3),  # A
    ], dim=1)
    g_emp = torch.rand(B, H, W, K) * 1e-3
    mask  = torch.ones(B, H, W)
    mask[:, ::3, ::3] = 0.0

    taus = [1, 2, 4, 8, 16, 32, 48, 64]
    print(f"v3.2 linear : {PhysicsReconLoss(taus)(preds, g_emp, mask).item():.6e}")
    print(f"v3.2 log    : {PhysicsReconLoss(taus, log_space=True)(preds, g_emp, mask).item():.6e}")
    print(f"v3.1 legacy : {PhysicsReconLoss(taus, legacy_2ch=True)(preds[:,:2], g_emp, mask).item():.6e}")
