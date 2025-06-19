import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentAct(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, eps=1e-5, momentum=0.1, **kwargs):
        """
        Args:
            alpha (float): Parameter for skewness.
            beta (float): Parameter for kurtosis.
            gamma (float): Parameter for scaling.
            eps (float): Small value to avoid division by zero.
            momentum (float): Momentum for running statistics.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.eps   = eps
        # running stats for inference
        self.register_buffer("running_mu",  torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("running_S",   torch.zeros(1))
        self.register_buffer("running_K",   torch.ones(1)*3)

        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mu  = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            sigma = torch.sqrt(var + self.eps)
            z = (x - mu) / sigma

            S = (z**3).mean(dim=0, keepdim=True)
            K = (z**4).mean(dim=0, keepdim=True)

            # update running stats
            self.running_mu  = (1-self.momentum)*self.running_mu + self.momentum*mu.detach()
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var.detach()
            self.running_S   = (1-self.momentum)*self.running_S  + self.momentum*S.detach()
            self.running_K   = (1-self.momentum)*self.running_K  + self.momentum*K.detach()
        else:
            mu  = self.running_mu
            var = self.running_var
            sigma = torch.sqrt(var + self.eps)
            z = (x - mu) / sigma
            S = self.running_S
            K = self.running_K

        g = torch.sigmoid(self.alpha*S + self.beta*(K-3))
        y = z + self.gamma * g * torch.tanh(z**3)
        return y
