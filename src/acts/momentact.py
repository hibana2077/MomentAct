import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentAct(nn.Module):
    def __init__(self, channels=32, eps=1e-5, momentum=0.1, inplace=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("s_moving", torch.zeros(channels))
        self.register_buffer("k_moving", torch.zeros(channels))
        
        # 可學習映射參數
        self.γ_skew = nn.Parameter(torch.ones(channels))
        self.β_skew = nn.Parameter(torch.zeros(channels))
        self.γ_kurt = nn.Parameter(torch.ones(channels))
        self.β_kurt = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # 計算高階矩 (沿批次+空間維)
        dims = tuple(range(1, x.ndim))  # 保留通道維
        mu = x.mean(dim=dims, keepdim=True)
        x_centered = x - mu
        var = x_centered.var(dim=dims, keepdim=True, unbiased=False) + self.eps
        
        # 計算偏度/峰度 (無偏估計)
        B = x.numel() // x.size(1)
        skew = (x_centered.pow(3).mean(dim=dims, keepdim=True) * (B/(B-1))) / var.pow(1.5)
        kurt = (x_centered.pow(4).mean(dim=dims, keepdim=True) * (B/(B-1))) / var.pow(2) - 3
        
        # 訓練/測試模式切換
        if self.training:
            # 更新移動平均
            self.s_moving = (1-self.momentum)*self.s_moving + self.momentum*skew.squeeze()
            self.k_moving = (1-self.momentum)*self.k_moving + self.momentum*kurt.squeeze()
        else:
            skew = self.s_moving.view_as(skew)
            kurt = self.k_moving.view_as(kurt)
        
        # 動態參數生成 (梯度分離)
        skew_det = skew.detach()
        kurt_det = kurt.detach()
        α = (self.γ_skew.view_as(skew)*skew_det + self.β_skew.view_as(skew)).clamp(-1,1)
        β = (self.γ_kurt.view_as(kurt)*kurt_det + self.β_kurt.view_as(kurt)).relu()
        
        # 混合激活
        return β * torch.tanh(α * x) + (1 - β) * F.gelu(x)