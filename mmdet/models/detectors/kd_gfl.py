
### Added by hakk ###

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from ..utils import multi_apply, unpack_gt_instances
from .kd_single_stage import KDSingleStageDetector
import pdb

@MODELS.register_module()
class KDGFL(KDSingleStageDetector):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds = \
            multi_apply(self.forward_kd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds = \
            multi_apply(self.forward_kd_single, stu_x,
                        self.bbox_head.scales, module=self)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

    def forward_kd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            feats: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl,\
            new_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.bbox_head.prior_generator.strides,
                avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses

