import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

class FocalLoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, alpha=0.25, gamma=2, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
