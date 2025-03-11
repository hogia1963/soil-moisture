import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, l1_weight=0.5, ssim_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def ssim_loss(self, x, y):
        # Simplified SSIM loss
        c1 = (0.01 * 2) ** 2
        c2 = (0.03 * 2) ** 2
        
        mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
        
        sigma_x = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        sigma_y = torch.var(y, dim=(2, 3), keepdim=True, unbiased=False)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2))
        
        return 1 - ssim.mean()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.l1_weight * l1_loss + 
                     self.ssim_weight * ssim_loss)
        
        return {
            'total': total_loss,
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item()
        }
