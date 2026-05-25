import torch
import torch.nn as nn


class PhysicsReconLoss(nn.Module):
    """
    Self-supervised Physics Reconstruction loss.

    Three modes (set exactly one flag):

    shape_only=True  [v3.4 default]
        Target g_empirical is already per-pixel normalised (starts at 1 for
        each cell pixel, 0 for background/held-out). Only G_theory is
        normalised here:
            G_theory_norm(τ) = (1+γ) / (1+γτ^α)
        Loss = MSE(G_theory_norm, g_empirical_norm).
        No amplitude in the model; no mean-regression attractor.

    legacy_2ch=True  [v3.1 style]
        Both G_theory and g_empirical normalised inside the loss.
        Use when the raw (un-normalised) G_empirical is passed as target.

    (default)  [v3.2/v3.3]
        Amplitude-aware: preds has 3 channels [γ,α,A].
        G_theory = A / (1+γτ^α) compared directly to raw g_empirical.
    """

    def __init__(self, recon_taus, shape_only=False, legacy_2ch=False,
                 log_space=False):
        super().__init__()
        taus = torch.as_tensor(recon_taus, dtype=torch.float32)
        self.register_buffer("taus", taus.view(1, 1, 1, -1))
        self.shape_only = shape_only
        self.legacy_2ch = legacy_2ch
        self.log_space  = log_space

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
            # Target pre-normalised by dataset → only normalise theory
            g_theory = g_theory / (g_theory[..., 0:1] + eps)
            # g_empirical unchanged (already starts at 1 per cell pixel)

        elif self.legacy_2ch:
            # Normalise both sides inside the loss
            g_theory    = g_theory    / (g_theory[..., 0:1] + eps)
            g_empirical = g_empirical / (g_empirical[..., 0:1].abs() + eps)

        else:
            # Amplitude-aware mode (v3.2/v3.3): A/(1+γτ^α)
            amp = preds[:, 2].unsqueeze(-1)
            g_theory = amp / (1.0 + gamma * torch.pow(self.taus, alpha))

        if self.log_space:
            sq_err = (torch.log(g_theory.clamp(min=1e-8))
                      - torch.log(g_empirical.abs().clamp(min=1e-8))) ** 2
        else:
            sq_err = (g_theory - g_empirical) ** 2

        mask4d  = train_mask.unsqueeze(-1)
        n_terms = mask4d.sum() * self.taus.numel() + 1e-10
        return (sq_err * mask4d).sum() / n_terms


if __name__ == "__main__":
    import torch
    B, H, W, K = 2, 64, 64, 8
    taus = [1, 2, 4, 8, 16, 32, 48, 64]
    mask = torch.ones(B, H, W); mask[:, ::3, ::3] = 0.0

    # v3.4: 2-ch preds, pre-normalised target (starts at 1)
    p2   = torch.cat([torch.rand(B,1,H,W)*0.5, torch.rand(B,1,H,W)*2], dim=1)
    g_norm = torch.rand(B, H, W, K); g_norm[..., 0] = 1.0
    print(f"v3.4 shape_only : {PhysicsReconLoss(taus, shape_only=True)(p2, g_norm, mask).item():.4e}")

    # v3.1 legacy: raw g_empirical
    g_raw = torch.rand(B, H, W, K) * 1e-3
    print(f"v3.1 legacy_2ch : {PhysicsReconLoss(taus, legacy_2ch=True)(p2, g_raw, mask).item():.4e}")

    # v3.2 amplitude mode
    p3 = torch.cat([p2, torch.full((B,1,H,W), 1e-3)], dim=1)
    print(f"v3.2 amplitude  : {PhysicsReconLoss(taus)(p3, g_raw, mask).item():.4e}")
