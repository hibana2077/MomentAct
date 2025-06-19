import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.activations import swish, mish, gelu, hard_swish

# ------------------ MomentMixAct ------------------ #
class MomentMixAct(nn.Module):
    """
    Dynamic mixture of {ReLU, LeakyReLU, GELU, Swish}
    using channel-wise skewness & kurtosis as soft selectors.
    """
    def __init__(self, eps=1e-5, momentum=0.1, leak=0.05, **kwargs):
        super().__init__()
        self.eps, self.momentum, self.leak = eps, momentum, leak
        # learnable α, β (1×4 vector)
        self.alpha = nn.Parameter(torch.zeros(4))
        self.beta = nn.Parameter(torch.zeros(4))
        # running stats for inference
        self.register_buffer("running_S", torch.zeros(1))
        self.register_buffer("running_K", torch.ones(1) * 3)
        # pre-compute sqrt eps for efficiency
        self.sqrt_eps = eps ** 0.5

    def _moments(self, x):
        # x: (B,C,H,W) - compute over spatial dims
        dims = (0, 2, 3) if x.dim() == 4 else (0,)
        
        # Single pass for mean and variance
        mu = x.mean(dim=dims, keepdim=True)
        x_centered = x - mu
        var = x_centered.pow(2).mean(dim=dims, keepdim=True)
        
        # Standardize once
        z = x_centered / (var + self.eps).sqrt()
        
        # Compute higher moments efficiently
        z2 = z.pow(2)
        S = (z * z2).mean(dim=dims, keepdim=True).squeeze()
        K = (z2 * z2).mean(dim=dims, keepdim=True).squeeze()
        
        return S, K

    def __repr__(self):
        """Print current running statistics in repr format"""
        return (f"{self.__class__.__name__}(eps={self.eps}, momentum={self.momentum}, "
                f"leak={self.leak}, running_S={self.running_S.item():.6f}, "
                f"running_K={self.running_K.item():.6f}, "
                f"alpha={self.alpha.tolist()}, beta={self.beta.tolist()})")

    def forward(self, x):
        if self.training:
            S, K = self._moments(x)
            # Update running stats without gradient
            S_mean, K_mean = S.mean().detach(), K.mean().detach()
            self.running_S.mul_(1 - self.momentum).add_(S_mean, alpha=self.momentum)
            self.running_K.mul_(1 - self.momentum).add_(K_mean, alpha=self.momentum)
        else:
            S_mean, K_mean = self.running_S, self.running_K

        # Compute weights efficiently
        logits = self.alpha * S_mean + self.beta * (K_mean - 3)
        w = F.softmax(logits, dim=0)

        # Apply activations - vectorized computation
        activations = torch.stack([
            F.relu(x),
            mish(x),
            gelu(x),
            swish(x),
        ])  # (4, B, C, H, W)
        
        # Weighted sum
        return torch.sum(w.view(-1, 1, 1, 1, 1) * activations, dim=0)