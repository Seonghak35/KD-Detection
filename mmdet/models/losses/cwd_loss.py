#####################
### Added by hakk ###
#####################

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss

import pdb

@weighted_loss
def cwd_loss(pred, target, tau):
    assert pred.shape[-2:] == target.shape[-2:]
    N, C, H, W = pred.shape

    softmax_target = F.softmax(target.view(-1, W*H) / tau, dim=1)

    logsoftmax = torch.nn.LogSoftmax(dim=1)
    loss = torch.sum(softmax_target*logsoftmax(target.view(-1, W*H) / tau) - 
                    softmax_target*logsoftmax(pred.view(-1, W*H) / tau)) * (
                    tau**2)

    return loss / (C*N)


@MODELS.register_module()
class CWDLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, tau=1.0):
        super(CWDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.tau = tau

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * cwd_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, tau=self.tau)
        return loss
