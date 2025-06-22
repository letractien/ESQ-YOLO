# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist
from .nwd import NWDLoss


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, use_nwd=False, nwd_ratio=0.5, eps=1e-7, constant=12.8):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.use_nwd = use_nwd
        self.nwd_ratio = nwd_ratio
        self.nwd_loss = NWDLoss(eps=eps, constant=constant)
        
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        ciou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_ciou = 1.0 - ciou

        if self.use_nwd:
            nwd = self.nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False).unsqueeze(-1)
            loss_nwd = 1.0 - nwd
            loss_iou = ((1.0 - self.nwd_ratio) * loss_ciou + self.nwd_ratio * loss_nwd) * weight
            loss_iou = loss_iou.sum() / target_scores_sum
        else:
            loss_iou = (loss_ciou * weight).sum() / target_scores_sum

        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
