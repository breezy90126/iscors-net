import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Masked MSE loss between predicted parameter maps and sparse GT.
    When a mask is provided only the masked pixels contribute to the loss,
    which is essential for sparse supervision (e.g. center 3x3 only).
    """
    def __init__(self, lambda_gamma=1.0, lambda_alpha=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_gamma = lambda_gamma
        self.lambda_alpha = lambda_alpha

    def forward(self, preds, targets, mask=None):
        """
        Args:
            preds:   (B, 2, H, W) — channel 0 is Gamma, 1 is Alpha.
            targets: (B, 2, H, W) — GT maps.
            mask:    (B, H, W) float, 1 where supervised, 0 elsewhere.
                     If None, uses standard MSE over the full patch.
        """
        gamma_pred = preds[:, 0]
        alpha_pred = preds[:, 1]
        gamma_gt   = targets[:, 0]
        alpha_gt   = targets[:, 1]

        if mask is not None:
            n_valid = mask.sum() + 1e-10
            loss_gamma = ((gamma_pred - gamma_gt) ** 2 * mask).sum() / n_valid
            loss_alpha = ((alpha_pred - alpha_gt) ** 2 * mask).sum() / n_valid
        else:
            loss_gamma = self.mse(gamma_pred, gamma_gt)
            loss_alpha = self.mse(alpha_pred, alpha_gt)

        total_loss = self.lambda_gamma * loss_gamma + self.lambda_alpha * loss_alpha
        return total_loss, loss_gamma, loss_alpha


if __name__ == "__main__":
    preds   = torch.rand(4, 2, 64, 64)
    targets = torch.rand(4, 2, 64, 64)
    mask    = torch.zeros(4, 64, 64)
    mask[:, 30:33, 30:33] = 1.0  # center 3x3

    criterion = PhysicsInformedLoss(lambda_gamma=1.0, lambda_alpha=2.0)

    total_no_mask, lg, la = criterion(preds, targets)
    print(f"No mask  — Total: {total_no_mask:.4f}, Gamma: {lg:.4f}, Alpha: {la:.4f}")

    total_masked, lg_m, la_m = criterion(preds, targets, mask)
    print(f"Masked   — Total: {total_masked:.4f},  Gamma: {lg_m:.4f}, Alpha: {la_m:.4f}")
