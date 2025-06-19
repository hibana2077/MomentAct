import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.activations import swish, mish, gelu, hard_swish

# ------------------ MomentMixAct ------------------ #
class MomentMixAct(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.alpha = nn.Parameter(torch.zeros(4))   # (4,)
        self.beta  = nn.Parameter(torch.zeros(4))   # (4,)
        self.register_buffer("running_S", torch.zeros(1))  # 之後會改成 (C,)
        self.register_buffer("running_K", torch.ones(1)*3)

    @staticmethod
    def _moments(x, eps):
        dims = (0, 2, 3)            # B,H,W
        mu  = x.mean(dims, keepdim=True)
        var = x.var(dims, unbiased=False, keepdim=True) + eps
        z   = (x - mu) * var.rsqrt()
        S   = (z.pow(3)).mean(dims) # 形狀 (C,)
        K   = (z.pow(4)).mean(dims) # 形狀 (C,)
        return S, K                 # 都與 x 保持梯度

    def __repr__(self):
        """Print current running statistics in repr format"""
        return (f"{self.__class__.__name__}(eps={self.eps}, momentum={self.momentum}, "
                f"alpha={self.alpha.tolist()}, beta={self.beta.tolist()})")

    def forward(self, x):
        S, K = self._moments(x, self.eps)           # (C,)
        if self.training:
            # running stats 也改成 (C,) 以便推論時仍能 per-channel
            if self.running_S.numel() != S.numel():
                self.running_S = torch.zeros_like(S)
                self.running_K = torch.ones_like(K)
            self.running_S.lerp_(S.detach(), self.momentum)
            self.running_K.lerp_(K.detach(), self.momentum)
            S_use, K_use = S, K
        else:
            S_use, K_use = self.running_S, self.running_K

        # 4×C logits，不要 detach，保持可微
        logits = (self.alpha.view(4, 1) * S_use +
                  self.beta.view(4, 1)  * (K_use - 3))
        w = F.softmax(logits, dim=0)                # 形狀 (4,C)

        # 算四種 activation
        out = w[0].unsqueeze(1).unsqueeze(2) * F.relu(x)
        out += w[1].unsqueeze(1).unsqueeze(2) * F.leaky_relu(x, 0.05)
        out += w[2].unsqueeze(1).unsqueeze(2) * F.gelu(x)
        out += w[3].unsqueeze(1).unsqueeze(2) * swish(x)
        return out