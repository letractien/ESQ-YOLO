import torch
import torch.nn as nn


class NWDLoss(nn.Module):
    def __init__(self, eps=1e-7, constant=12.8):
        super().__init__()
        self.eps = eps
        self.constant = constant

    def xyxy_to_xywh(self, boxes):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack((cx, cy, w, h), dim=1)

    def wasserstein_loss(self, pred, target, xywh):
        if not xywh:
            pred = self.xyxy_to_xywh(pred)
            target = self.xyxy_to_xywh(target)

        center1 = pred[:, :2]
        center2 = target[:, :2]

        whs = center1[:, :2] - center2[:, :2]
        center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + self.eps

        w1 = pred[:, 2]  + self.eps
        h1 = pred[:, 3]  + self.eps
        w2 = target[:, 2] + self.eps
        h2 = target[:, 3] + self.eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        wasserstein_2 = center_distance + wh_distance
        return torch.exp(-torch.sqrt(wasserstein_2) / self.constant)

    def forward(self, pred_boxes, target_boxes, xywh=False):
        loss = self.wasserstein_loss(pred_boxes, target_boxes, xywh)
        return loss
    
