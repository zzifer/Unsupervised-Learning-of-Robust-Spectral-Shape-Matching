import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


# 计算两个张量在指定维度上的点积
def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    # torch.mul(X, Y) 计算逐元素乘积，然后 torch.sum(..., dim=dim) 在指定维度上求和，最终得到点积结果
    return torch.sum(torch.mul(X, Y), dim=dim)


@LOSS_REGISTRY.register()
class DirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, L):
        assert feats.dim() == 3

        if self.normalize:
            # 对特征进行归一化
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, torch.bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return self.loss_weight * loss
