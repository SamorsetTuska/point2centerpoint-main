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


def clean_rank_list(rank_list):
    # 清洗rank_list中的空tensor
    new_rank_list = []
    for rank in rank_list:
        if rank.numel() > 0:
            new_rank_list.append(rank)
    return new_rank_list

def get_local_rank(quality, indices):
    # 对于每个gt，将其对应的box按quality排序并赋予序号返回
    bs = len(indices)
    device = quality.device
    tgt_size = [len(tgt_ind) for _, tgt_ind in indices]  # 每张图预测数量
    ind_start = 0
    rank_list = []
    for i in range(bs):
        if  tgt_size[i] == 0:
            rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
            continue
        num_tgt = max(indices[i][1]) + 1  # 此单张图像的tgt数量
        # split quality of one item
        quality_per_img = quality[ind_start:ind_start+tgt_size[i]]  # 图像中所有t得分的质量
        ind_start += tgt_size[i]  # 更新到下一张图索引
        # suppose candidate bag sizes are equal
        # k = torch.div(tgt_size[i], num_tgt, rounding_mode='floor')  # 每个gt有多少个box，实际上ota的K是动态的，应当采用其他方法
        # sort quality in each candidate bag
        '''
        quality_per_img = quality_per_img.reshape(num_tgt, k)
        ind = quality_per_img.sort(dim=-1,descending=True)[1]
        # scatter ranks, eg:[0.3,0.6,0.5] -> [2,0,1]
        rank_per_img = torch.zeros_like(quality_per_img, dtype=torch.long, device = device)
        rank_per_img.scatter_(-1, ind, torch.arange(k,device=device, dtype=torch.long).repeat(num_tgt,1))
        rank_list.append(rank_per_img.flatten())
        '''
        sort_per_img = torch.zeros_like(quality_per_img)
        # gts = []
        for gt in range(num_tgt):
            ins = torch.nonzero(indices[i][1] == gt).squeeze()  # 所有该gt的box索引
            # gts.append(ins.size())
            quality_gt = quality_per_img[ins]
            quality_gt_sorted_ins = torch.argsort(quality_gt)  # 对这些box排序
            sort_per_img[ins] = quality_gt_sorted_ins.type(torch.float32)
        rank_list.append(sort_per_img.type(torch.float32))
    rank_list = clean_rank_list(rank_list)
    for r in range(len(rank_list)):
        assert rank_list[r].dtype == torch.float32, '{}中的{}类型有误'.format(rank_list, rank_list[r])
    return torch.cat((rank_list), dim=0)

class SetCriterion_align(nn.Module):
    """ This class computes the loss for SparseRCNN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, grid_size, pc_range):
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
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.use_ota = cfg.USE_OTA
        self.use_aux = cfg.USE_AUX
        self.alpha = cfg.ALPHA
        self.gamma = cfg.GAMMA


    def sigmoid_focal_align_loss(self, pos_logits, pos_labels, neg_logits, neg_labels, alpha=-1, gamma=2,
                               reduction="none", ):
        neg_p_t = torch.sigmoid(neg_logits.float())
        neg_labels = neg_labels.float()
        pos_p_t = torch.sigmoid(pos_logits.float())
        pos_labels = pos_labels.float()**(1 - alpha) * pos_p_t ** alpha
        pos_labels = torch.clamp(pos_labels, 0.01).detach()
        neg_ce_loss = F.binary_cross_entropy(neg_p_t, neg_labels, reduction="none")
        neg_ce_loss = neg_ce_loss * (neg_p_t ** gamma)
        pos_ce_loss = F.binary_cross_entropy(pos_p_t, pos_labels, reduction="none")
        # pos_ce_loss = pos_ce_loss * ((pos_labels - pos_p_t) ** gamma) * alpha
        loss = torch.cat([pos_ce_loss, neg_ce_loss], dim=0)

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
        #probs = torch.softmax(src_logits, dim=-1)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  ###0315
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_boxes = outputs['pred_boxes'][idx]
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
        IoU = iou3d_nms_utils.boxes_iou_bev(src_boxes, bev_tgt_bbox).diag().squeeze(0)
        # wrong = target_classes_o[IoU==0]
        # IoU = (IoU * IoU) / (IoU * IoU).max()

        alpha = 0.25
        gamma = 2.0
        tau = 1.5
        #pos_weights = torch.zeros_like(src_logits)
        #neg_weights = prob ** gamma
        #t = target_classes_o ** alpha * IoU ** (1 - alpha)
        #rank = get_local_rank(t, indices)  # 计算每个gt对应box的排序
        #w_rank = torch.exp(-rank/tau)
        #t = t * w_rank  # 将IoU信息加入label_loss中
        #warm_up = False
        #if torch.mean(IoU) < 0.1:  # IoU太低时也不能作为参考
        #    warm_up = True
        src_logits = src_logits.flatten(0, 1)  # 展平为BN*3
        # probs = probs.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]  # 有gt的inds
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1  # fg=1, bg=0,0,0

        l = labels.size()[-1]
        class_loss = 0
        for i in range(l):
            each_src_logits = src_logits[:, i]  # 单独一类的logits
            label = labels[:, i]  # 单独一类的label
            prob = each_src_logits.sigmoid()
            #prob = probs[:, i]  # 单独一类的prob
            prob = torch.clamp(prob, min=1e-6, max=1-(1e-6))
            pos_weights = torch.zeros_like(label)
            neg_weights = prob ** gamma
            # neg_weights = torch.ones_like(label)
            all_gt_label = label[pos_inds]
            gt_label = all_gt_label == 1  # 是否是此类目标
            #iou = torch.masked_select(IoU, gt_label)  # 只保留此类目标的IoU
            iou = IoU[gt_label].detach()  # 只保留此类目标的IoU
            # 只保留indices中的此类目标
            indice = []

            start_inds = 0
            for img in indices:
                g_b = list(img)  # gt-box对
                num_gts = len(g_b[0])
                mask = gt_label[start_inds: start_inds+num_gts]  # 用mask筛掉其他类别的目标
                g_b[0] = g_b[0][mask]
                g_b[1] = g_b[1][mask]
                g_b = tuple(g_b)
                start_inds += num_gts
                indice.append(g_b)

            # import pdb;pdb.set_trace()
            t = prob[pos_inds][gt_label]**alpha * iou**(1-alpha)
            t = torch.clamp(t, 0.01).detach()
            # t = iou ** alpha
            if self.use_ota:
                rank = get_local_rank(t, indice)
                rank_weight = torch.exp(-rank/tau)
                t = t * rank_weight

            gt_indice = torch.nonzero(gt_label).flatten()
            gt_inds = pos_inds[gt_indice]
            pos_weights[gt_inds] = t
            neg_weights[gt_inds] = 1 - t
            gt_pos = pos_weights[pos_inds][gt_indice]

            loss = -1 * (pos_weights * prob.log() + neg_weights * (1-prob).log())  # 计算该类别的BCE
            num = len(pos_weights)
            loss = loss.sum() / num
            class_loss += loss
            '''
        labels[pos_inds, target_classes[pos_inds]] = IoU
        pos_logits = src_logits[pos_inds, :]
        pos_labels = labels[pos_inds, :]
        neg_logits = src_logits[target_classes == self.num_classes, :]
        neg_labels = labels[target_classes == self.num_classes, :]
        class_loss = self.sigmoid_focal_align_loss(pos_logits,
                                                 pos_labels,
                                                 neg_logits,
                                                 neg_labels,
                                                 alpha=self.alpha,
                                                 gamma=self.gamma,
                                                 reduction="sum") / num_boxes
            '''
        losses = {'loss_ce': class_loss}

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
        # test_ious = cal_iou_3d(src_boxes.reshape(1, -1, 7), bev_tgt_bbox.reshape(1, -1, 7))
        # test_iou = test_ious.sum()
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
            torch.distributed.all_reduce(num_boxes)  # 统计所有目标框数量
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
                # if self.use_ota:
                #     indices, num_gt_matched = self.matcher(aux_outputs, targets)
                #     num_boxes = num_gt_matched
                # else:
                #     indices = self.matcher(aux_outputs, targets)
                #     # indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'hm' or loss == 'density' or loss == 'mask':
                        continue

                    if loss == 'boxes':
                        kwargs = {'stage': i}  # 辅助头只考虑box回归

                    l_dict = self.get_loss_aux(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def add_sin_difference(boxes1, boxes2, dim=6):
    # 将rad_encoding嵌入box中
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


class HungarianMatcher_align(nn.Module):
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
        self.repeat_num = 2

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
        use_mto = outputs["use_mto"]
        C = []
        for b in range(bs):

            # We flatten to compute the cost matrices in a batch
            if self.use_focal:
                out_prob = outputs["pred_logits"][b, ...].unsqueeze(0).flatten(0, 1).sigmoid()
                # [batch_size * num_queries, num_classes]
            else:
                out_prob = outputs["pred_logits"][b, ...].unsqueeze(0).flatten(0, 1).softmax(-1)
                # [batch_size * num_queries, num_classes]

            out_bbox = outputs["pred_boxes_match"][b, ...].unsqueeze(0).flatten(0, 1)  # [batch_size * num_queries, 7]

            # Also concat the target labels and boxes

            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["gt_boxes"]

            image_size_tgt = targets[b]["image_size_xyxy_tgt"]
            offset = targets[b]["offset_size_xyxy_tgt"]
            grid_size_tgt = targets[b]["grid_size_tgt"]
            grid_size_out = targets[b]["grid_size_xyz"].unsqueeze(0).unsqueeze(1).repeat(1, num_queries, 1).flatten(0,
                                                                                                                    1)

            out_bbox_ = torch.ones_like(out_bbox, dtype=out_bbox.dtype)
            out_bbox_[:, :3] = out_bbox[:, :3] / grid_size_out
            out_bbox_[:, 3:6] = out_bbox[:, 3:6] / grid_size_out
            # the recommended principal range is between [-180, 180) degrees
            out_bbox_[:, -1] = out_bbox[:, -1]

            tgt_bbox_ = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
            tgt_bbox_[:, :3] = (tgt_bbox[:, :3] - offset) / image_size_tgt
            tgt_bbox_[:, 3:6] = (tgt_bbox[:, 3:6]) / image_size_tgt
            tgt_bbox_[:, -1] = tgt_bbox[:, -1]
            tgt_bbox_sin = torch.sin(tgt_bbox_[:, -1])
            tgt_bbox_cos = torch.cos(tgt_bbox_[:, -1])
            tgt_bbox_angle = torch.cat([tgt_bbox_sin.unsqueeze(-1), tgt_bbox_cos.unsqueeze(-1)], dim=-1)

            cost_bbox = torch.cdist(out_bbox_[:, :-1], tgt_bbox_[:, :-1], p=1)

            code_weight = torch.tensor([0.5, 0.5], device=cost_bbox.device)
            out_bbox_angle = outputs['pred_rot'][b, ...].unsqueeze(0).flatten(0, 1)
            cost_r = torch.cdist(out_bbox_angle * code_weight, tgt_bbox_angle * code_weight, p=1)

            cost_bbox += cost_r

            bev_tgt_bbox = torch.ones_like(tgt_bbox, dtype=tgt_bbox.dtype)
            bev_tgt_bbox[:, :3] = tgt_bbox_[:, :3] * grid_size_tgt
            bev_tgt_bbox[:, 3:6] = tgt_bbox_[:, 3:6] * grid_size_tgt
            bev_tgt_bbox[:, -1] = tgt_bbox[:, -1]

            ious = box_utils.boxes3d_nearest_bev_iou(out_bbox, bev_tgt_bbox)
            # # ious = iou3d_nms_utils.boxes_iou3d_gpu(out_bbox, bev_tgt_bbox)
            cost_giou = -ious

            # out_bbox = out_bbox.unsqueeze(1).repeat(1, bev_tgt_bbox.shape[0], 1)
            # bev_tgt_bbox = bev_tgt_bbox.unsqueeze(0).repeat(out_bbox.shape[0], 1, 1)
            # DIoU = 1 - cal_diou_3d(out_bbox, bev_tgt_bbox)
            # cost_giou = -DIoU
            #
            # diou_match = ((DIoU.max(dim=1)[0] + 1) / 2 + 1e-6).sqrt().unsqueeze(-1).repeat(1, out_prob.shape[1])

            # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # # but approximate it in 1 - proba[target class].
            # # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal:
                # Compute the classification cost.
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                out_prob = out_prob[:, tgt_ids]
                t = out_prob ** alpha * ious**(1 - alpha)
                t = torch.clamp(t, 1e-6).detach()
                # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-6).log())
                # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-6).log())
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * t * (-(1 - out_prob + 1e-6).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (1 - t) * (-(out_prob + 1e-6).log())

                # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # [N,M]
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Final cost matrix
            Cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            if use_mto:
                Cost = Cost.repeat(1, self.repeat_num)

            C.append(Cost)
        indices = [linear_sum_assignment(c.cpu()) for i, c in enumerate(C)]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices
