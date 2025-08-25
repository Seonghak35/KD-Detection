#####################
### Added by hakk ###
#####################

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@MODELS.register_module()
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
