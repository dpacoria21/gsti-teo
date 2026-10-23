"""Opt-in uncertain-region supervision for torchvision Mask R-CNN.

Derived from torchvision 0.26 RPN/RoIHeads sampling and loss contracts (BSD-3):
https://github.com/pytorch/vision/tree/v0.26.0/torchvision/models/detection
Copyright (c) Soumith Chintala 2016. The complete upstream license, conditions,
and disclaimer are preserved in LICENSES/torchvision-BSD-3-Clause.txt.
Local adaptations add uncertain-region sampling and masked loss; they are not
an upstream torchvision implementation or an endorsement by its contributors.
No global functions/classes are patched. Evaluation delegates to torchvision,
and state-dict keys/shapes stay unchanged. The dataset must exclude whole GT
instances intersecting uncertain pixels before constructing boxes.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, project_masks_on_boxes
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import roi_align


def boxes_touch_ignore(boxes: torch.Tensor, supervision_mask: torch.Tensor) -> torch.Tensor:
    """Conservative pixel intersection, including fractional box boundaries.

    Any uncertain pixel touched by a candidate's bounding box disqualifies that
    candidate as a background example. Positive candidates still derive from
    complete trusted GT. With ignore present, candidates entirely in padding
    have no supervised support and are also excluded. Fully supervised inputs
    retain torchvision's original treatment of image padding.
    """
    if supervision_mask.dtype != torch.bool or supervision_mask.ndim != 2:
        raise ValueError("supervision_mask debe ser bool[H,W].")
    if supervision_mask.all():
        return torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    height, width = supervision_mask.shape
    corners = torch.cat((boxes[:, :2].floor(), boxes[:, 2:].ceil()), dim=1).long()
    x1, y1, x2, y2 = corners.unbind(1)
    x1, x2 = x1.clamp(0, width), x2.clamp(0, width)
    y1, y2 = y1.clamp(0, height), y2.clamp(0, height)
    integral = F.pad((~supervision_mask).to(torch.int64).cumsum(0).cumsum(1), (1, 0, 1, 0))
    count = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return (count > 0) | (x2 <= x1) | (y2 <= y1)


class IgnoreTransform(GeneralizedRCNNTransform):
    """Resize the supervision map in the same image coordinate system as GT."""

    def resize(self, image, target=None):
        mask = target.get("supervision_mask") if target is not None else None
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(image.shape[-2:])):
            raise ValueError("supervision_mask debe ser bool[H,W] y coincidir con la imagen.")
        image, target = super().resize(image, target)
        if mask is not None:
            target["supervision_mask"] = F.interpolate(mask[None, None].float(), size=image.shape[-2:], mode="nearest")[
                0, 0
            ].bool()
        return image, target


class IgnoreRegionProposalNetwork(RegionProposalNetwork):
    """Do not sample uncertain anchors as negative training examples."""

    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched_boxes = super().assign_targets_to_anchors(anchors, targets)
        for boxes, target, assigned in zip(anchors, targets, labels):
            if "supervision_mask" in target:
                uncertain = boxes_touch_ignore(boxes, target["supervision_mask"])
                assigned[(assigned == 0) & uncertain] = -1
        return labels, matched_boxes

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # The stock reduction divides by zero when every anchor is ignored.
        if not any((assigned >= 0).any() for assigned in labels):
            return objectness.sum() * 0, pred_bbox_deltas.sum() * 0
        return super().compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)


def supervised_mask_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs, supervision_masks):
    """BCE only at RoI samples fully supported by annotated pixels.

    Use the same RoIAlign coordinate convention and sampling as GT mask targets.
    Fractional valid support is excluded, preventing uncertain pixels near a
    boundary from contributing a negative target through interpolation.
    """
    size = mask_logits.shape[-1]
    labels = torch.cat([label[index] for label, index in zip(gt_labels, matched_idxs)])
    targets = torch.cat(
        [
            project_masks_on_boxes(mask, boxes, index, size)
            for mask, boxes, index in zip(gt_masks, proposals, matched_idxs)
        ]
    )
    if targets.numel() == 0:
        return mask_logits.sum() * 0
    supported = []
    for boxes, valid in zip(proposals, supervision_masks):
        rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), dim=1)
        support = roi_align(valid[None, None].to(boxes), rois, (size, size), 1.0)[:, 0]
        supported.append(support >= 1.0 - 1e-6)
    supported = torch.cat(supported)
    selected_logits = mask_logits[torch.arange(len(labels), device=labels.device), labels]
    if not supported.any():
        return selected_logits.sum() * 0
    return F.binary_cross_entropy_with_logits(selected_logits[supported], targets[supported])


class IgnoreRoIHeads(RoIHeads):
    """Training-only sampler/loss changes; inference follows stock RoIHeads."""

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        if targets is None:
            raise ValueError("Se requieren targets para entrenar.")
        dtype, device = proposals[0].dtype, proposals[0].device
        gt_boxes = [target["boxes"].to(dtype) for target in targets]
        gt_labels = [target["labels"] for target in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        for boxes, target, assigned in zip(proposals, targets, labels):
            if "supervision_mask" in target:
                uncertain = boxes_touch_ignore(boxes, target["supervision_mask"])
                assigned[(assigned == 0) & uncertain] = -1
        sampled = self.subsample(labels)
        matched_boxes = []
        for i, indices in enumerate(sampled):
            proposals[i], labels[i], matched_idxs[i] = (
                proposals[i][indices],
                labels[i][indices],
                matched_idxs[i][indices],
            )
            boxes = gt_boxes[i]
            if boxes.numel() == 0:
                boxes = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_boxes.append(boxes[matched_idxs[i]])
        regression_targets = self.box_coder.encode(matched_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training or targets is None or not any("supervision_mask" in t for t in targets):
            return super().forward(features, proposals, image_shapes, targets)
        if not all("supervision_mask" in target for target in targets):
            raise ValueError("Cada target del lote opt-in necesita supervision_mask.")
        for target in targets:
            if not target["boxes"].is_floating_point() or target["labels"].dtype != torch.int64:
                raise TypeError("Los targets necesitan boxes flotantes y labels int64.")
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        if sum(len(label) for label in labels):
            classifier_loss, box_loss = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        else:
            classifier_loss, box_loss = class_logits.sum() * 0, box_regression.sum() * 0
        positives = [torch.where(label > 0)[0] for label in labels]
        mask_proposals = [boxes[index] for boxes, index in zip(proposals, positives)]
        mask_indices = [matched[index] for matched, index in zip(matched_idxs, positives)]
        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_logits = self.mask_predictor(self.mask_head(mask_features))
        mask_loss = supervised_mask_loss(
            mask_logits,
            mask_proposals,
            [t["masks"] for t in targets],
            [t["labels"] for t in targets],
            mask_indices,
            [t["supervision_mask"] for t in targets],
        )
        return [], {"loss_classifier": classifier_loss, "loss_box_reg": box_loss, "loss_mask": mask_loss}


def enable_training_ignore(model):
    """Install local subclasses once, retaining the existing parameter objects.

    Call before constructing the optimizer. No checkpoint tensor, global class,
    or other model instance changes. Only the Mask R-CNN architecture is supported.
    """
    if isinstance(model.roi_heads, IgnoreRoIHeads):
        return model
    rpn, roi, transform = model.rpn, model.roi_heads, model.transform
    if (
        type(rpn) is not RegionProposalNetwork
        or type(roi) is not RoIHeads
        or type(transform) is not GeneralizedRCNNTransform
    ):
        raise TypeError("El opt-in requiere RPN, RoIHeads y transform estándar de torchvision.")
    if not roi.has_mask() or roi.has_keypoint():
        raise TypeError("El opt-in solo admite Mask R-CNN sin keypoints.")
    replacement_rpn = IgnoreRegionProposalNetwork(
        rpn.anchor_generator,
        rpn.head,
        rpn.proposal_matcher.high_threshold,
        rpn.proposal_matcher.low_threshold,
        rpn.fg_bg_sampler.batch_size_per_image,
        rpn.fg_bg_sampler.positive_fraction,
        dict(rpn._pre_nms_top_n),
        dict(rpn._post_nms_top_n),
        rpn.nms_thresh,
        rpn.score_thresh,
    )
    replacement_rpn.box_coder = rpn.box_coder
    replacement_rpn.box_similarity = rpn.box_similarity
    replacement_rpn.min_size = rpn.min_size
    replacement_roi = IgnoreRoIHeads(
        roi.box_roi_pool,
        roi.box_head,
        roi.box_predictor,
        roi.proposal_matcher.high_threshold,
        roi.proposal_matcher.low_threshold,
        roi.fg_bg_sampler.batch_size_per_image,
        roi.fg_bg_sampler.positive_fraction,
        roi.box_coder.weights,
        roi.score_thresh,
        roi.nms_thresh,
        roi.detections_per_img,
        roi.mask_roi_pool,
        roi.mask_head,
        roi.mask_predictor,
    )
    replacement_transform = IgnoreTransform(
        transform.min_size,
        transform.max_size,
        transform.image_mean,
        transform.image_std,
        size_divisible=transform.size_divisible,
        fixed_size=transform.fixed_size,
        _skip_resize=transform._skip_resize,
    )
    # Keep parent mode without resetting individually frozen child modules.
    replacement_rpn.training, replacement_roi.training = rpn.training, roi.training
    replacement_transform.training = transform.training
    model.rpn, model.roi_heads, model.transform = replacement_rpn, replacement_roi, replacement_transform
    return model
