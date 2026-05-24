import torch
import torch.nn as nn


class PhysicsReconLoss(nn.Module):
    """
    Self-supervised Physics Reconstruction loss for iSCORS internal learning.

    Given predicted (γ, α) per pixel, compute the theoretical autocorrelation
        G_theory(τ; γ, α) = 1 / (1 + γ · τ^α)
    and compare against the precomputed empirical G_empirical(τ) per pixel.

    Loss applied only at supervised pixels (train_mask = 1); held-out pixels
    are excluded so we retain a generalisation test set.

    Args:
        recon_taus: 1D tensor of τ lags used for reconstruction (e.g. [1,2,4,8,16,32,48,64]).
    """

    def __init__(self, recon_taus):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        # Shape (1, 1, 1, K) so it broadcasts over (B, H, W, K)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))

    def forward(self, preds, g_empirical, train_mask):
        """
        Args:
            preds:       (B, 2, H, W) — channel 0 γ, channel 1 α.
            g_empirical: (B, H, W, K) — precomputed empirical G(τ) per pixel.
            train_mask:  (B, H, W)    — 1.0 at supervised pixels, 0.0 at held-out.

        Returns:
            loss  : scalar.
        """
        gamma = preds[:, 0].unsqueeze(-1)            # (B, H, W, 1)
        alpha = preds[:, 1].unsqueeze(-1)            # (B, H, W, 1)

        # Differentiable G_theory; taus≥1 so no division-by-zero risk.
        g_theory = 1.0 / (1.0 + gamma * torch.pow(self.taus, alpha))   # (B, H, W, K)

        sq_err = (g_theory - g_empirical) ** 2        # (B, H, W, K)
        mask4d = train_mask.unsqueeze(-1)             # (B, H, W, 1)
        n_terms = mask4d.sum() * self.taus.numel() + 1e-10
        loss = (sq_err * mask4d).sum() / n_terms
        return loss


if __name__ == "__main__":
    B, H, W, K = 2, 64, 64, 8
    preds = torch.rand(B, 2, H, W)
    g_emp = torch.rand(B, H, W, K) * 0.1
    mask  = torch.ones(B, H, W)
    mask[:, ::3, ::3] = 0.0  # ~11% held-out

    loss_fn = PhysicsReconLoss(recon_taus=[1, 2, 4, 8, 16, 32, 48, 64])
    loss = loss_fn(preds, g_emp, mask)
    print(f"Loss: {loss.item():.6f}")
