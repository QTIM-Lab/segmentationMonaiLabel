import torch

# Hi

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Assuming logits are the raw model outputs and targets are the ground truth masks
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        
        # p = probs.view(num, -1)
        p = probs.reshape(num, -1)
        t = targets.view(num, -1)
        
        intersection = torch.einsum('bi,bi->b', [p, t])
        p_sum = torch.einsum('bi->b', [p])
        t_sum = torch.einsum('bi->b', [t])
        
        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
        loss = 1 - dice.mean()
        
        return loss
