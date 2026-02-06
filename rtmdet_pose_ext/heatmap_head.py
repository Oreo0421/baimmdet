# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class HeatmapHead(nn.Module):
    """
    Simple heatmap-based keypoint head.

    Design:
    - Predict ONE set of keypoints per image (from heatmap argmax).
    - If detector outputs multiple bboxes (N>1), keep only TOP-1 bbox (by score),
      and attach this one set of keypoints to that bbox.
    - Coordinate restore: heatmap coords -> resized img coords -> original coords.

    Output format for MMPose CocoMetric compatibility:
      - pred_instances.keypoints:       [N, K, 2]  (x, y)
      - pred_instances.keypoint_scores: [N, K]
    """

    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        loss_keypoint: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.conv1 = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_channels)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feat_channels)
        self.pred_layer = nn.Conv2d(feat_channels, num_keypoints, kernel_size=1)

        self.loss_keypoint = MODELS.build(loss_keypoint) if loss_keypoint is not None else None

    def forward(self, feats: Tuple[Tensor, ...]) -> Tensor:
        """Return heatmaps [B, K, H, W]. Uses feats[0] by default."""
        x = feats[0]  # [B, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        heatmaps = self.pred_layer(x)
        return heatmaps

    def loss(self, feats: Tuple[Tensor, ...], batch_data_samples: List) -> Dict[str, Tensor]:
        """Compute keypoint heatmap loss if loss_keypoint is configured."""
        heatmaps = self.forward(feats)  # [B, K, H, W]
        B, K, H, W = heatmaps.shape

        if self.loss_keypoint is None:
            return {}

        gt_heatmaps = []
        target_weights = []

        for ds in batch_data_samples:
            # gt heatmap
            if hasattr(ds, "gt_keypoints_heatmap"):
                hm = ds.gt_keypoints_heatmap
            else:
                hm = heatmaps.new_zeros((K, H, W))

            if isinstance(hm, torch.Tensor):
                hm_t = hm.to(device=heatmaps.device, dtype=heatmaps.dtype)
            else:
                hm_t = heatmaps.new_tensor(hm, dtype=heatmaps.dtype, device=heatmaps.device)
            gt_heatmaps.append(hm_t)

            # target weight from visibility (v>0), shape [K]
            if hasattr(ds, "gt_keypoints"):
                kpts = ds.gt_keypoints
                if isinstance(kpts, torch.Tensor) and kpts.numel() >= K * 3:
                    v = kpts[:, 2].to(device=heatmaps.device, dtype=heatmaps.dtype)
                    w = (v > 0).to(dtype=heatmaps.dtype)
                else:
                    w = heatmaps.new_ones((K,))
            else:
                w = heatmaps.new_ones((K,))
            target_weights.append(w)

        gt_heatmaps = torch.stack(gt_heatmaps, dim=0)        # [B, K, H, W]
        target_weights = torch.stack(target_weights, dim=0)  # [B, K]

        try:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps, target_weights)
        except TypeError:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps)

        return {"loss_keypoint": loss_kpt}

    @torch.no_grad()
    def predict(
        self,
        feats: Tuple[Tensor, ...],
        batch_results,
        batch_data_samples=None,
        rescale: bool = True,
    ):
        """
        Attach keypoints to detection results.

        Args:
            feats: tuple of feature maps
            batch_results: usually List[InstanceData] from det head,
                           or List[DetDataSample]-like with .pred_instances
            batch_data_samples: List[DetDataSample] providing metainfo
            rescale: restore to original image coords if True

        Returns:
            batch_results (updated in place)
        """
        heatmaps = self.forward(feats)  # [B, K, H, W]
        batch_keypoints, batch_scores = self._decode_heatmap(heatmaps)  # [B,K,2], [B,K]

        hm_h, hm_w = heatmaps.shape[-2], heatmaps.shape[-1]

        # ----- coordinate restore: heatmap -> img_shape -> ori_shape -----
        if batch_data_samples is not None:
            for b, ds in enumerate(batch_data_samples):
                meta = getattr(ds, "metainfo", None)
                if meta is None:
                    continue

                img_shape = meta.get("img_shape", None)  # resized shape
                ori_shape = meta.get("ori_shape", None)  # original shape
                if img_shape is None or ori_shape is None:
                    continue

                img_h, img_w = int(img_shape[0]), int(img_shape[1])
                ori_h, ori_w = int(ori_shape[0]), int(ori_shape[1])

                # heatmap -> resized image (pixel coords)
                batch_keypoints[b, :, 0] = (batch_keypoints[b, :, 0] + 0.5) * (float(img_w) / float(hm_w)) - 0.5
                batch_keypoints[b, :, 1] = (batch_keypoints[b, :, 1] + 0.5) * (float(img_h) / float(hm_h)) - 0.5

                # resized -> original
                if rescale and ori_w > 0 and ori_h > 0:
                    w_scale = float(img_w) / float(ori_w)
                    h_scale = float(img_h) / float(ori_h)
                    if w_scale > 0:
                        batch_keypoints[b, :, 0] = batch_keypoints[b, :, 0] / w_scale
                    if h_scale > 0:
                        batch_keypoints[b, :, 1] = batch_keypoints[b, :, 1] / h_scale

        # ----- attach predictions -----
        for i, item in enumerate(batch_results):
            keypoints = batch_keypoints[i]  # [K,2]
            scores = batch_scores[i]        # [K]

            # Case A: batch_results[i] is InstanceData
            if isinstance(item, InstanceData):
                if len(item) == 0:
                    item.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    item.keypoint_scores = scores.new_zeros((0, self.num_keypoints))
                    continue

                # pick top-1 by detection score if exists
                if hasattr(item, "scores") and item.scores is not None and len(item.scores) > 0:
                    top1 = int(item.scores.argmax())
                else:
                    top1 = 0

                item = item[top1:top1 + 1]   # slice WHOLE InstanceData -> keeps all fields consistent
                batch_results[i] = item

                item.keypoints = keypoints.unsqueeze(0)        # [1, K, 2]
                item.keypoint_scores = scores.unsqueeze(0)     # [1, K]
                continue

            # Case B: DetDataSample-like with .pred_instances
            if hasattr(item, "pred_instances"):
                pred = item.pred_instances
                if len(pred) == 0:
                    pred.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    pred.keypoint_scores = scores.new_zeros((0, self.num_keypoints))
                    continue

                if hasattr(pred, "scores") and pred.scores is not None and len(pred.scores) > 0:
                    top1 = int(pred.scores.argmax())
                else:
                    top1 = 0

                pred = pred[top1:top1 + 1]
                item.pred_instances = pred

                pred.keypoints = keypoints.unsqueeze(0)        # [1, K, 2]
                pred.keypoint_scores = scores.unsqueeze(0)     # [1, K]
                continue

            # Unknown type: do nothing

        return batch_results

    def _decode_heatmap(self, heatmaps: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode heatmaps by argmax.

        Returns:
            keypoints: [B, K, 2] in heatmap coords (x, y)
            scores:    [B, K] max heatmap value
        """
        B, K, H, W = heatmaps.shape
        heatmaps_ = heatmaps.view(B, K, -1)          # [B, K, H*W]
        maxvals, idx = torch.max(heatmaps_, dim=2)   # [B, K], [B, K]

        x = (idx % W).to(dtype=heatmaps.dtype)
        y = (idx // W).to(dtype=heatmaps.dtype)
        keypoints = torch.stack([x, y], dim=2)       # [B, K, 2]
        scores = maxvals
        return keypoints, scores

