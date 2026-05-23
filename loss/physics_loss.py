import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Computes the loss between the predicted parameter maps and the 
    Ground Truth maps derived from the 1% sparse spatial calculation.
    """
    def __init__(self, lambda_gamma=1.0, lambda_alpha=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_gamma = lambda_gamma
        self.lambda_alpha = lambda_alpha

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (B, 2, H, W). Channel 0 is Gamma, 1 is Alpha.
            targets: Tensor of shape (B, 2, H, W) containing the 1% GT values.
        """
        gamma_pred = preds[:, 0, :, :]
        alpha_pred = preds[:, 1, :, :]
        
        gamma_gt = targets[:, 0, :, :]
        alpha_gt = targets[:, 1, :, :]
        
        # Calculate individual MSE losses
        loss_gamma = self.mse(gamma_pred, gamma_gt)
        loss_alpha = self.mse(alpha_pred, alpha_gt)
        
        # Total loss
        total_loss = (self.lambda_gamma * loss_gamma) + (self.lambda_alpha * loss_alpha)
        
        return total_loss, loss_gamma, loss_alpha

if __name__ == "__main__":
    # Test the loss
    preds = torch.rand(4, 2, 64, 64)
    targets = torch.rand(4, 2, 64, 64)
    
    criterion = PhysicsInformedLoss(lambda_gamma=1.0, lambda_alpha=2.0)
    total, lg, la = criterion(preds, targets)
    
    print(f"Total Loss: {total.item():.4f}")
    print(f"Gamma Loss: {lg.item():.4f}, Alpha Loss: {la.item():.4f}")
