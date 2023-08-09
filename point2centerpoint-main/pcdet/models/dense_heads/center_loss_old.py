#
# Modified by HJH
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from ...utils.misc import get_world_size, is_dist_avail_and_initialized

from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import pdb
from ...ops.iou3d_nms import iou3d_nms_utils
from ..model_utils import centernet_utils
from ...utils import box_utils, loss_utils, common_utils
from ...ops.Rotated_IoU.oriented_iou_loss import cal_diou_3d, cal_iou_3d
import copy


class SetCriterion(nn.Module):
    """ This class computes the loss for SparseRCNN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal, grid_size, pc_range):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.back_ground_index = 1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.use_ota = cfg.USE_OTA
        self.use_aux = cfg.USE_AUX

        if self.use_focal:
            self.focal_loss_alpha = cfg.ALPHA
            self.focal_loss_gamma = cfg.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def sigmoid_focal_iou_loss(seld, inputs, targets, iou, alpha=-1, gamma=2, reduction="none"):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        iou = iou.unsqueeze(1).repeat(1, 1, inputs.shape[-1]).view(-1, inputs.shape[-1])
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(p * iou, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  ###0315
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_boxes = outputs['pred_boxes'][idx]
        src_boxes_angle = outputs['pred_rot'][idx]
        target_boxes = torch.cat([t['gt_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        assert len(targets) > 0
        grid_size_out = targets[0]['grid_size_xyz'].unsqueeze(0).repeat(len(target_boxes), 1)
        image_size_xyxy_tgt = targets[0]['image_size_xyxy_tgt'][0].unsqueeze(0).repeat(len(target_boxes), 1)
        offset_size_xyxy_tgt = targets[0]['offset_size_xyxy_tgt'][0].unsqueeze(0).repeat(len(target_boxes), 1)

        src_boxes_ = torch.ones_like(src_boxes, dtype=src_boxes.dtype)
        src_boxes_[:, :3] = src_boxes[:, :3] / grid_size_out
        src_boxes_[:, 3:6] = src_boxes[:, 3:6] / grid_size_out
        src_boxes_[:, -1] = src_boxes[:, -1]

        target_boxes_ = torch.ones_like(target_boxes, dtype=target_boxes.dtype)
        target_boxes_[:, :3] = (target_boxes[:, :3] - offset_size_xyxy_tgt) / image_size_xyxy_tgt
        target_boxes_[:, 3:6] = target_boxes[:, 3:6] / image_size_xyxy_tgt
        target_boxes_[:, -1] = target_boxes[:, -1]

        bev_tgt_bbox = torch.ones_like(target_boxes, dtype=target_boxes.dtype)
        bev_tgt_bbox[:, :3] = target_boxes_[:, :3] * grid_size_out
        bev_tgt_bbox[:, 3:6] = target_boxes_[:, 3:6] * grid_size_out
        bev_tgt_bbox[:, -1] = target_boxes[:, -1]

        DIoUs = ((1 - cal_diou_3d(src_boxes.reshape(1, -1, 7), bev_tgt_bbox.reshape(1, -1, 7)) + 1) / 2 + 1e-6).sqrt()
        iou = torch.ones(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
        iou[idx] = DIoUs.squeeze(0)

        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1  # fg=1
            # comp focal loss.
            # class_loss = sigmoid_focal_loss_jit(
            #     src_logits,
            #     labels,
            #     alpha=self.focal_loss_alpha,
            #     gamma=self.focal_loss_gamma,
            #     reduction="sum",
            # ) / num_boxes
            class_loss = self.sigmoid_focal_iou_loss(src_logits,
                                                     labels,
                                                     iou,
                                                     alpha=self.focal_loss_alpha,
                                                     gamma=self.focal_loss_gamma,
                                                     reduction="sum") / num_boxes
            losses = {'loss_ce': class_loss}
            # assert not torch.any(torch.isnan(src_logits))
            # assert not torch.any(torch.isnan(labels))
            #
            # assert not torch.any(torch.isnan(class_loss))
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, stage=5):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        src_boxes_angle = outputs['pred_rot'][idx]

        target_boxes = torch.cat([t['gt_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        assert len(targets) > 0
        grid_size_out = targets[0]['grid_size_xyz'].unsqueeze(0).repeat(len(target_boxes), 1)
        image_size_xyxy_tgt = targets[0]['image_size_xyxy_tgt'][0].unsqueeze(0).repeat(len(target_boxes), 1)
        offset_size_xyxy_tgt = targets[0]['offset_size_xyxy_tgt'][0].unsqueeze(0).repeat(len(target_boxes), 1)

        src_boxes_ = torch.ones_like(src_boxes, dtype=src_boxes.dtype)
        src_boxes_[:, :3] = src_boxes[:, :3] / grid_size_out
        src_boxes_[:, 3:6] = src_boxes[:, 3:6] / grid_size_out
        src_boxes_[:, -1] = src_boxes[:, -1]

        target_boxes_ = torch.ones_like(target_boxes, dtype=target_boxes.dtype)
        target_boxes_[:, :3] = (target_boxes[:, :3] - offset_size_xyxy_tgt) / image_size_xyxy_tgt
        target_boxes_[:, 3:6] = target_boxes[:, 3:6] / image_size_xyxy_tgt
        target_boxes_[:, -1] = target_boxes[:, -1]
        target_boxes_sin = torch.sin(target_boxes_[:, -1]).unsqueeze(-1)
        target_boxes_cos = torch.cos(target_boxes_[:, -1]).unsqueeze(-1)

        code_weight = torch.as_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5], device=target_boxes_.device)

        box_preds_reg = torch.cat([src_boxes_[:, :6], src_boxes_angle], dim=-1)
        box_preds_reg = box_preds_reg * code_weight
        target_boxes_reg = torch.cat([target_boxes_[:, :6], target_boxes_sin, target_boxes_cos], dim=-1)
        target_boxes_reg = target_boxes_reg * code_weight
        # loss_bbox = F.l1_loss(box_preds_reg, target_boxes_reg, reduction='none')
        loss_bbox = F.smooth_l1_loss(box_preds_reg, target_boxes_reg, reduction='none')

        # loss_bbox_presudo = F.l1_loss(presudo_boxes * code_weight, targets_presudo * code_weight, reduction='none')

        bev_tgt_bbox = torch.ones_like(target_boxes, dtype=target_boxes.dtype)
        bev_tgt_bbox[:, :3] = target_boxes_[:, :3] * grid_size_out
        bev_tgt_bbox[:, 3:6] = target_boxes_[:, 3:6] * grid_size_out
        bev_tgt_bbox[:, -1] = target_boxes[:, -1]

        ious = cal_diou_3d(src_boxes.reshape(1, -1, 7), bev_tgt_bbox.reshape(1, -1, 7))

        # src_boxes_iou = outputs['iou']
        # target_iou = cal_iou_3d(src_boxes.reshape(1, -1, 7), bev_tgt_bbox.reshape(1, -1, 7))
        # target_iou_o = torch.full(src_boxes_iou.shape[:2], -1.0,
        #                           dtype=torch.float, device=src_boxes_iou.device)
        # target_iou_o[idx] = target_iou * 2 - 1
        # loss_iou = F.l1_loss(src_boxes_iou, target_iou_o.unsqueeze(-1), reduction='mean')

        loss_giou = ious
        # loss_giou = ious + loss_iou

        loss_bbox = loss_bbox.sum() / num_boxes

        # loss_bbox_presudo = loss_bbox_presudo.sum() / num_presudo_boxes
        # assert not torch.any(torch.isnan(loss_bbox))

        losses['loss_bbox'] = loss_bbox
        # losses['loss_bbox_presudo'] = loss_bbox_presudo

        loss_giou = loss_giou.sum() / num_boxes
        losses['loss_giou'] = loss_giou

        if outputs['vel'] is not None:
            src_boxes_vel = outputs['vel'][idx]
            target_boxes_vel = torch.cat([t['gt_vel'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_vel = F.l1_loss(src_boxes_vel, target_boxes_vel, reduction='none')
            loss_vel = loss_vel.sum() / num_boxes
            losses['loss_vel'] = loss_vel

        return losses

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def neg_loss_cornernet(self, pred, gt, mask=None):
        """
        Refer to https://github.com/tianweiy/CenterPoint.
        Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
        Args:
            pred: (batch x c x h x w)
            gt: (batch x c x h x w)
            mask: (batch x h x w)
        Returns:
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        if mask is not None:
            mask = mask[:, None, :, :].float()
            pos_loss = pos_loss * mask
            neg_loss = neg_loss * mask
            num_pos = (pos_inds.float() * mask).sum()
        else:
            num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def loss_hm(self, outputs, targets, indices, num_boxes):
        outputs['heatmaps_pred'] = self.sigmoid(outputs['heatmaps_pred'])
        loss_hm = self.neg_loss_cornernet(outputs['heatmaps_pred'], targets[0]['heatmaps'])
        losses = {'loss_hm': loss_hm}

        return losses

    def loss_density(self, outputs, targets, indices, num_boxes):
        loss_mse = nn.MSELoss(size_average=False).cuda()
        loss_density = loss_mse(outputs['density_map_pred'], targets[0]['density_maps'])
        losses = {'loss_density': loss_density}

        return losses

    def loss_mask(self, outputs, targets, indices, num_boxes):
        loss_mask = F.cross_entropy(outputs['mask_maps'], targets[0]['mask_maps'].long())
        losses = {'loss_mask': loss_mask}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):

        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'hm': self.loss_hm,
            # 'mask': self.loss_mask
            # 'density': self.loss_density
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_aux(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.use_ota:
            indices, num_gt_matched = self.matcher(outputs_without_aux, targets)
            num_boxes = num_gt_matched
        else:
            indices = self.matcher(outputs_without_aux, targets)
            # indices, match_num_per_gt = self.matcher(outputs_without_aux, targets)
            # num_boxes *= match_num_per_gt

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_ota:
                    indices, num_gt_matched = self.matcher(aux_outputs, targets)
                    num_boxes = num_gt_matched
                else:
                    indices = self.matcher(outputs_without_aux, targets)
                    # indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'hm' or loss == 'density' or loss == 'mask':
                        continue

                    if loss == 'boxes':
                        kwargs = {'stage': i}

                    l_dict = self.get_loss_aux(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def add_sin_difference(boxes1, boxes2, dim=6):
    assert dim != -1
    rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
    rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
    boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
    return boxes1, boxes2


def add_sin_difference_pairwise(boxes1, boxes2):
    '''
    boxes1: N,1
    boxes2: M,1

    return: NxM
    sin(a-b)=sinacosb-cosasinb
    '''
    r_sin_dist = torch.zeros(boxes1.shape[0], boxes2.shape[0], dtype=torch.float, device=boxes1.device)
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            r_sin_dist[i, j] = torch.sin(boxes1[i, 0]) * torch.cos(boxes2[j, 0]) - torch.cos(boxes1[i, 0]) * torch.sin(
                boxes2[j, 0])

    return r_sin_dist


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_center: float = 1,
                 use_focal: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cfg = cfg
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_center = cost_center
        self.use_focal = use_focal

        if self.use_focal:
            self.focal_loss_alpha = cfg.ALPHA
            self.focal_loss_gamma = cfg.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes_match"].flatten(0, 1)  # [batch_size * num_queries, 7]
        #
        # if outputs['cur_epoch'] > 60:
        #     out_iou = ((outputs["iou"] + 1) * 0.5).flatten(0, 1).repeat(1, 3)  # [batch_size * num_queries, 1]
        #     out_iou = out_iou ** torch.as_tensor([0.68, 0.71, 0.65], device=out_iou.device)
        #     out_prob = out_prob ** torch.as_tensor([0.32, 0.29, 0.35], device=out_prob.device)
        #     out_prob = (out_iou.clamp(0, 1) * out_prob).clamp(0, 1)
        # out_bbox = outputs["anchor_points"].flatten(0, 1)  # [batch_size * num_queries, 3]

        # Also concat the target labels and boxes

        tgt_ids = torch.cat([v["labels"] for v in targets])  ####0315
        tgt_bbox = torch.cat([v["gt_boxes"] for v in targets])

        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        offset = torch.cat([v["offset_size_xyxy_tgt"] for v in targets])
        grid_size_tgt = torch.cat([v["grid_size_tgt"] for v in targets])

        grid_size_out = torch.cat([v["grid_size_xyz"].unsqueeze(0) for v in targets])
        grid_size_out = grid_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)

        out_bbox_ = torch.ones_like(out_bbox, dtype=out_bbox.dtype)
        out_bbox_[:, :3] = out_bbox[:, :3] / grid_size_out
        out_bbox_[:, 3:6] = out_bbox[:, 3:6] / grid_size_out
        # the recommended principal range is between [-180, 180) degrees
        out_bbox_[:, -1] = out_bbox[:, -1]

        tgt_bbox_ = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
        tgt_bbox_[:, :3] = (tgt_bbox[:, :3] - offset) / image_size_tgt
        tgt_bbox_[:, 3:6] = (tgt_bbox[:, 3:6]) / image_size_tgt
        # tgt_bbox_[:, -1] = -tgt_bbox[:, -1]
        tgt_bbox_[:, -1] = tgt_bbox[:, -1]
        tgt_bbox_sin = torch.sin(tgt_bbox_[:, -1])
        tgt_bbox_cos = torch.cos(tgt_bbox_[:, -1])
        tgt_bbox_angle = torch.cat([tgt_bbox_sin.unsqueeze(-1), tgt_bbox_cos.unsqueeze(-1)], dim=-1)

        # grid_anchor = outputs["grid_anchor"].flatten(0, 1) / grid_size_out

        # cost_bbox = torch.cdist(grid_anchor, tgt_bbox_[:, :3], p=2)
        # cost_bbox = torch.cdist(out_bbox_[:, :2], tgt_bbox_[:, :2], p=1)
        # cost_center = torch.cdist(out_bbox_[:, :3], tgt_bbox_[:, :3], p=2)
        # cost_bbox += 0.2 * torch.cdist(out_bbox_[:, :-1], tgt_bbox_[:, :-1], p=1)
        cost_bbox = torch.cdist(out_bbox_[:, :-1], tgt_bbox_[:, :-1], p=1)
        # cost_bbox += 0.2 * torch.cdist(out_bbox_, tgt_bbox_, p=1)
        # cost_r = 0.2 * iou3d_nms_utils.sin_difference_pairwise(out_bbox_[:, 6:7], tgt_bbox_[:, 6:7])
        # cost_r = iou3d_nms_utils.sin_difference_pairwise(out_bbox_[:, 6:7], tgt_bbox_[:, 6:7])
        code_weight = torch.tensor([0.5, 0.5], device=cost_bbox.device)
        cost_r = torch.cdist(outputs['pred_rot'].flatten(0, 1) * code_weight, tgt_bbox_angle * code_weight, p=1)
        # cost_r = torch.cdist(outputs['pred_rot'].flatten(0, 1), tgt_bbox_angle, p=1)

        cost_bbox += cost_r

        bev_tgt_bbox = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
        bev_tgt_bbox[:, :3] = tgt_bbox_[:, :3] * grid_size_tgt
        bev_tgt_bbox[:, 3:6] = tgt_bbox_[:, 3:6] * grid_size_tgt
        # bev_tgt_bbox[:, -1] = -tgt_bbox[:, -1]
        bev_tgt_bbox[:, -1] = tgt_bbox[:, -1]

        ious = box_utils.boxes3d_nearest_bev_iou(out_bbox, bev_tgt_bbox)
        # ious = iou3d_nms_utils.boxes_iou3d_gpu(out_bbox, bev_tgt_bbox)

        cost_giou = -ious

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-6).log())
            # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-6).log())
            s = (ious * ious).max(dim=1)[0]
            s = s / s.max()
            s = s.unsqueeze(-1).repeat(1, out_prob.shape[1])
            pos_cost_class = alpha * ((s - out_prob) ** gamma) * (
                -((s * out_prob + 1e-6).log() + ((1 - s) * (1 - out_prob) + 1e-6).log()))
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # [N,M]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Final cost matrix

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["gt_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # for i in range(len(indices)):
        #     mask = outputs["saved_index"][:, 0] == i
        #     saved_ind = outputs["saved_index"][mask, :][:, -1]
        #     for j in indices[i][0]:
        #         assert j in saved_ind

        return indices
    # def forward(self, outputs, targets):
    #     """ Performs the matching
    #
    #     Params:
    #         outputs: This is a dict that contains at least these entries:
    #              "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
    #              "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
    #
    #         targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
    #              "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
    #                        objects in the target) containing the class labels
    #              "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
    #
    #     Returns:
    #         A list of size batch_size, containing tuples of (index_i, index_j) where:
    #             - index_i is the indices of the selected predictions (in order)
    #             - index_j is the indices of the corresponding selected targets (in order)
    #         For each batch element, it holds:
    #             len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    #     """
    #     indexes_repeat = []
    #     cur_epoch = outputs['cur_epoch']
    #     if cur_epoch < 10:
    #         match_num_per_gt = 9
    #     else:
    #         match_num_per_gt = 1
    #
    #     bs, num_queries = outputs["pred_logits"].shape[:2]
    #     # We flatten to compute the cost matrices in a batch
    #
    #     if self.use_focal:
    #         out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
    #     else:
    #         out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
    #
    #     outputs_pred_box_clone = outputs["pred_boxes_match"].detach().clone()
    #     for i in range(match_num_per_gt):
    #
    #         out_bbox = outputs_pred_box_clone.flatten(0, 1)  # [batch_size * num_queries, 7]
    #         # out_bbox = outputs["anchor_points"].flatten(0, 1)  # [batch_size * num_queries, 3]
    #         # Also concat the target labels and boxes
    #         tgt_ids = torch.cat([v["labels"] for v in targets])  ####0315
    #         tgt_bbox = torch.cat([v["gt_boxes"] for v in targets])
    #         # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    #         # but approximate it in 1 - proba[target class].
    #         # The 1 is a constant that doesn't change the matching, it can be ommitted.
    #         if self.use_focal:
    #             # Compute the classification cost.
    #             alpha = self.focal_loss_alpha
    #             gamma = self.focal_loss_gamma
    #             neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-6).log())
    #             pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-6).log())
    #             cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # [N,M]
    #         else:
    #             cost_class = -out_prob[:, tgt_ids]
    #
    #         image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
    #
    #         offset = torch.cat([v["offset_size_xyxy_tgt"] for v in targets])
    #         grid_size_tgt = torch.cat([v["grid_size_tgt"] for v in targets])
    #
    #         grid_size_out = torch.cat([v["grid_size_xyz"].unsqueeze(0) for v in targets])
    #         grid_size_out = grid_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
    #
    #         out_bbox_ = torch.ones_like(out_bbox, dtype=out_bbox.dtype)
    #         out_bbox_[:, :3] = out_bbox[:, :3] / grid_size_out
    #         out_bbox_[:, 3:6] = out_bbox[:, 3:6] / grid_size_out
    #         # the recommended principal range is between [-180, 180) degrees
    #         out_bbox_[:, -1] = out_bbox[:, -1]
    #
    #         tgt_bbox_ = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
    #         tgt_bbox_[:, :3] = (tgt_bbox[:, :3] - offset) / image_size_tgt
    #         tgt_bbox_[:, 3:6] = (tgt_bbox[:, 3:6]) / image_size_tgt
    #         tgt_bbox_[:, -1] = tgt_bbox[:, -1]
    #         tgt_bbox_sin = torch.sin(tgt_bbox_[:, -1])
    #         tgt_bbox_cos = torch.cos(tgt_bbox_[:, -1])
    #         tgt_bbox_angle = torch.cat([tgt_bbox_sin.unsqueeze(-1), tgt_bbox_cos.unsqueeze(-1)], dim=-1)
    #
    #         # grid_anchor = outputs["grid_anchor"].flatten(0, 1) / grid_size_out
    #
    #         cost_bbox = torch.cdist(out_bbox_[:, :-1], tgt_bbox_[:, :-1], p=1)
    #         # cost_bbox += 0.2 * torch.cdist(out_bbox_, tgt_bbox_, p=1)
    #         # cost_r = 0.2 * iou3d_nms_utils.sin_difference_pairwise(out_bbox_[:, 6:7], tgt_bbox_[:, 6:7])
    #         # cost_r = iou3d_nms_utils.sin_difference_pairwise(out_bbox_[:, 6:7], tgt_bbox_[:, 6:7])
    #         code_weight = torch.tensor([0.5, 0.5], device=cost_bbox.device)
    #         cost_r = torch.cdist(outputs['pred_rot'].flatten(0, 1) * code_weight, tgt_bbox_angle * code_weight, p=1)
    #
    #         cost_bbox += cost_r
    #
    #         bev_tgt_bbox = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
    #         bev_tgt_bbox[:, :3] = tgt_bbox_[:, :3] * grid_size_tgt
    #         bev_tgt_bbox[:, 3:6] = tgt_bbox_[:, 3:6] * grid_size_tgt
    #         bev_tgt_bbox[:, -1] = tgt_bbox[:, -1]
    #
    #         ious = box_utils.boxes3d_nearest_bev_iou(out_bbox, bev_tgt_bbox)
    #         # ious = iou3d_nms_utils.boxes_iou3d_gpu(out_bbox, bev_tgt_bbox)
    #
    #         cost_giou = -ious
    #
    #         # Final cost matrix
    #
    #         # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
    #         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
    #         C = C.view(bs, num_queries, -1).cpu()
    #
    #         sizes = [len(v["gt_boxes"]) for v in targets]
    #         indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    #         indexes_repeat.append(indices)
    #         for b in range(bs):
    #             for index in indices[b][0]:
    #                 outputs_pred_box_clone[b, index] = torch.tensor([0] * 7, dtype=torch.float32,
    #                                                                 device=outputs_pred_box_clone.device)
    #
    #     indices = []
    #
    #     for b in range(bs):
    #         pred_id = []
    #         targets_id = []
    #         for t in range(len(indexes_repeat)):
    #             pred_id.append(indexes_repeat[t][b][0])
    #             targets_id.append(indexes_repeat[t][b][1])
    #         indices.append((np.hstack(pred_id), np.hstack(targets_id)))
    #
    #     return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
    #             indices], match_num_per_gt


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1,
                 use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = cfg.USE_FED_LOSS
        self.ota_k = cfg.OTA_K
        self.center_radius = cfg.CENTER_RADIUS
        if self.use_focal:
            self.focal_loss_alpha = cfg.ALPHA
            self.focal_loss_gamma = cfg.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 7]
            else:
                out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 7]

            indices = []
            matched_ids = []
            matched_results = []

            grid_size_out = torch.cat([v["grid_size_xyz"].unsqueeze(0) for v in targets])[0].unsqueeze(0)
            grid_size_out = grid_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
            assert bs == len(targets)
            num_matched_gt = 0
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                offset = targets[batch_idx]["offset_size_xyxy_tgt"]
                image_size_tgt = targets[batch_idx]["image_size_xyxy_tgt"]
                grid_size_tgt = targets[batch_idx]["grid_size_tgt"]
                image_size_out = targets[batch_idx]["image_size_xyxy_tgt"][0].repeat(num_queries, 1)
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue

                out_bbox_ = torch.ones_like(bz_boxes, dtype=out_bbox.dtype)
                out_bbox_[:, :3] = bz_boxes[:, :3] / grid_size_out
                out_bbox_[:, 3:6] = bz_boxes[:, 3:6] / grid_size_out
                out_bbox_[:, -1] = bz_boxes[:, -1]

                # out_bbox_world is the out_boxes in real word coordinate
                out_bbox_world = torch.ones_like(out_bbox_, dtype=out_bbox_.dtype)
                out_bbox_world[:, :3] = out_bbox_[:, :3] * image_size_out
                out_bbox_world[:, 3:6] = out_bbox_[:, 3:6] * image_size_out
                out_bbox_world[:, -1] = out_bbox_[:, -1]

                bz_gtboxs = targets[batch_idx]['gt_boxes']  # [num_gt, 7] unnormalized (cx, cy, cz, w, l, h, theta)
                tgt_bbox_ = torch.ones_like(bz_gtboxs, dtype=bz_gtboxs.dtype)
                tgt_bbox_[:, :3] = (bz_gtboxs[:, :3] - offset) / image_size_tgt
                tgt_bbox_[:, 3:6] = bz_gtboxs[:, 3:6] / image_size_tgt
                tgt_bbox_[:, -1] = bz_gtboxs[:, -1]

                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(out_bbox_world, bz_gtboxs, expanded_strides=32)

                tgt_bbox_bev = torch.ones_like(tgt_bbox_, dtype=tgt_bbox_.dtype)
                tgt_bbox_bev[:, :3] = tgt_bbox_[:, :3] * grid_size_tgt
                tgt_bbox_bev[:, 3:6] = tgt_bbox_[:, 3:6] * grid_size_tgt
                tgt_bbox_bev[:, -1] = tgt_bbox_[:, -1]

                pair_wise_ious = iou3d_nms_utils.boxes_iou3d_gpu(bz_boxes, tgt_bbox_bev)

                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

                cost_giou = -pair_wise_ious

                # Final cost matrix
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (
                    ~is_in_boxes_and_center)
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                # indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])
                matched_result, gt_matched_num_batch = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                # indices.append(indices_batchi)
                # matched_ids.append(matched_qidx)
                matched_results.append(matched_result)
                num_matched_gt += gt_matched_num_batch

        # return indices, matched_ids
        return matched_results, num_matched_gt

    def bev_rbbox_to_corners(self, rbbox):
        # generate clockwise corners and rotate it clockwise
        cx = rbbox[:, 0]
        cy = rbbox[:, 1]
        x_d = rbbox[:, 3]
        y_d = rbbox[:, 4]
        angle = rbbox[:, 6]
        a_cos = torch.cos(angle)
        a_sin = torch.sin(angle)
        corners_x = torch.stack([-x_d / 2, -x_d / 2, x_d / 2, x_d / 2], dim=1)
        corners_y = torch.stack([-y_d / 2, y_d / 2, y_d / 2, -y_d / 2], dim=1)
        corners = torch.tensor([0] * 8, dtype=torch.float, device=rbbox.device).unsqueeze(0).repeat(rbbox.shape[0], 1)
        for i in range(4):
            corners[:, 2 * i] = a_cos * corners_x[:, i] + a_sin * corners_y[:, i] + cx
            corners[:, 2 * i + 1] = -a_sin * corners_x[:, i] + a_cos * corners_y[:, i] + cy
        return corners

    def point_in_quadrilateral(self, pt_x, pt_y, corners):
        ab0 = corners[:, 2] - corners[:, 0]
        ab1 = corners[:, 3] - corners[:, 1]

        ad0 = corners[:, 6] - corners[:, 0]
        ad1 = corners[:, 7] - corners[:, 1]

        ap0 = pt_x - corners[:, 0]
        ap1 = pt_y - corners[:, 1]

        abab = (ab0 * ab0 + ab1 * ab1).unsqueeze(0).repeat(pt_x.shape[0], 1)
        abap = ab0 * ap0 + ab1 * ap1
        adad = (ad0 * ad0 + ad1 * ad1).unsqueeze(0).repeat(pt_x.shape[0], 1)
        adap = ad0 * ap0 + ad1 * ap1
        point_in_matrix = (abab >= abap).long() + (abap >= 0).long() + (adad >= adap).long() + (adap >= 0).long()

        return point_in_matrix

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)
        anchor_center_z = boxes[:, 2].unsqueeze(1)
        # anchor_center_x_transform = (anchor_center_x - target_gts[:, 0]) * torch.cos(np.pi / 2 - target_gts[:, -1]) + (
        #         anchor_center_y - target_gts[:, 1]) * torch.sin(np.pi / 2 - target_gts[:, -1])
        # anchor_center_y_transform = (anchor_center_y - target_gts[:, 1]) * torch.cos(np.pi / 2 - target_gts[:, -1]) + (
        #         anchor_center_x - target_gts[:, 0]) * torch.sin(np.pi / 2 - target_gts[:, -1])

        # TODO: whether the center of each anchor is inside a gt box, need to be double checked.
        bev_corners = self.bev_rbbox_to_corners(target_gts)
        point_in_quadrilateral = self.point_in_quadrilateral(anchor_center_x, anchor_center_y, bev_corners)

        b_d = anchor_center_z > (target_gts[:, 2] - (target_gts[:, 5] / 2)).unsqueeze(0)
        b_u = anchor_center_z < (target_gts[:, 2] + (target_gts[:, 5] / 2)).unsqueeze(0)

        is_in_boxes = ((point_in_quadrilateral + b_d.long() + b_u.long()) == 6)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = self.center_radius

        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        target_gts[:, 3:6] = target_gts[:, 3:6] * center_radius
        bev_corners = self.bev_rbbox_to_corners(target_gts)
        point_in_quadrilateral = self.point_in_quadrilateral(anchor_center_x, anchor_center_y, bev_corners)
        b_d = anchor_center_z > (target_gts[:, 2] - (center_radius * target_gts[:, 5] / 2)).unsqueeze(0)
        b_u = anchor_center_z < (target_gts[:, 2] + (center_radius * target_gts[:, 5] / 2)).unsqueeze(0)

        is_in_centers = ((point_in_quadrilateral + b_d.long() + b_u.long()) == 6)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)  # 每个anchor匹配的gt数量

        if (anchor_matching_gt > 1).sum() > 0:  # 如果存在一个anchor匹配了多个gt
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)  # 找到这些anchor匹配的gt中，cost最小的那个
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():  # 如果存在一个gt没有匹配的anchor
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()  # 计算有多少个gt没有匹配的anchor
            matched_query_id = matching_matrix.sum(1) > 0  # 找到已经匹配了的anchor
            cost[matched_query_id] += 100000.0  # 给这些anchor一个很大的cost，避免他们被再次匹配
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)  # 没有被匹配的gt的id
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])  # 找到cost最小的anchor
                matching_matrix[:, gt_idx][pos_idx] = 1.0  # 将这个anchor匹配到这个gt
            if (matching_matrix.sum(1) > 1).sum() > 0:  # 如果存在一个anchor匹配了多个gt
                anchor_matching_gt = matching_matrix.sum(1)  # 更新每个anchor匹配的gt数量
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # 找到这些anchor匹配的gt中，cost最小的那个
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost
                # anchor_matching_gt = matching_matrix.sum(1)
                # index_anchor_repeat = torch.nonzero(anchor_matching_gt > 1).squeeze(1)
                # for i in index_anchor_repeat.tolist():
                #     torch.nonzero(matching_matrix[i, :]).squeeze(1)
                # cost[anchor_matching_gt > 1] += 100  # increase cost of other gt for these queries
                # cost[anchor_matching_gt > 1, cost_argmin,] -= 100

        # assert not (matching_matrix.sum(1) > 1).sum() == 0
        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        # cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        # matched_query_id = torch.min(cost, dim=0)[1]
        # matched_query_id=list({}.fromkeys(matched_query_id.tolist()).keys())
        # matching_matrix_numpy = matching_matrix.cpu().numpy()
        # matched_pair = torch.as_tensor(np.where(matching_matrix_numpy == 1.0))
        # matched_query_id = matched_pair[0, :].squeeze(0)
        # gt_indices = matched_pair[1, :].squeeze(0)
        # return (selected_query, gt_indices), matched_query_id
        selected_query = torch.nonzero((selected_query == True).long(), as_tuple=False).squeeze(1)
        #
        # return (matched_query_id, gt_indices), len(gt_indices)

        return (selected_query, gt_indices), len(gt_indices)
