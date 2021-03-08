import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit
from functools import partial
from detectron2.config import configurable
from .matcher import ATSSMatcher,PointMatcher

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)

        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid




from detectron2.utils.registry import Registry

REPPOINT_LOSS_REGISTRY = Registry("REPPOINT_LOSS")
REPPOINT_LOSS_REGISTRY.__doc__ = ""

def build_reppoint_loss(cfg):
    name = cfg.MODEL.REPPOINT_LOSS.NAME
    return REPPOINT_LOSS_REGISTRY.get(name)(cfg)


import functools

import torch.nn.functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    #minpro = torch.full_like(pred_sigmoid,0.1)
    #pred_sigmoid = torch.maximum(minpro,pred_sigmoid)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    weight = weight.repeat(80,1).permute(1,0)
    criterion = nn.BCEWithLogitsLoss(weight, reduction='mean',size_average=avg_factor)
    #print('pred:',pred.size())
    #print('target:',target.size())
    #print('weight:',weight.size())
    loss = criterion(
        pred, target) * focal_weight
    return loss.sum()


class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        one = torch.where(target==1)
        #print('one',one)
        #print('pred:',pred[one])
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:

            loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
            return loss_cls

        else:
            raise NotImplementedError
from torch.nn import SmoothL1Loss

@REPPOINT_LOSS_REGISTRY.register()
class ReppointLoss(nn.Module):
    @configurable
    def __init__(self,num_class,loss_cls,loss_bbox_init,loss_bbox_refine,
                 loss_bbox_init_weight, loss_bbox_refine_weight,
                point_strides = [8, 16, 32, 64, 128],
                 num_points = 9,
                point_base_scale = 4,
                 transform_method='moment',
                 moment_mul=0.01,):
        super().__init__()
        self.num_classes = num_class
        self.loss_bbox_init_weight = loss_bbox_init_weight
        self.loss_bbox_refine_weight = loss_bbox_refine_weight
        self.background_label = num_class
        self.num_points = num_points
        self.cls_out_channels = self.num_classes
        self.loss_cls = loss_cls
        self.loss_bbox_init = loss_bbox_init
        self.loss_bbox_refine = loss_bbox_refine


        self.point_base_scale = point_base_scale
        self.point_strides = point_strides

        self.point_generators = [PointGenerator() for _ in self.point_strides]
        self.init_assigner = PointMatcher()
        self.refine_assigner = ATSSMatcher()
        # use PseudoSampler when sampling is False

        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        pass

    @classmethod
    def from_config(cls,cfg):
        loss_cls_weight = cfg.MODEL.REPPOINT_LOSS.LOSS_CLS_WEIGHT
        loss_bbox_init_weight = cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_INIT_WEIGHT
        loss_bbox_refine_weight = cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_REFINE_WEIGHT
        loss_smooth_beta = cfg.MODEL.REPPOINT_LOSS.LOSS_SMOOTH_BETA
        loss_focal_alpha = cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_ALPHA
        loss_focal_gamma = cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_GAMMA
        loss_bbox_init = SmoothL1Loss(reduction='sum',beta=loss_smooth_beta)
        loss_cls = FocalLoss(use_sigmoid=True,
                             gamma=loss_focal_gamma,
                             alpha=loss_focal_alpha,
                             loss_weight=loss_cls_weight)
        loss_bbox_refine = SmoothL1Loss(reduction='sum',beta=loss_smooth_beta)
        ret = {
            "num_class":cfg.MODEL.REPPOINT_HEAD.NUM_CLASS,
            "loss_bbox_init":loss_bbox_init,
            "loss_bbox_refine":loss_bbox_refine,
            "loss_bbox_init_weight":loss_bbox_init_weight,
            "loss_bbox_refine_weight":loss_bbox_refine_weight,
            "loss_cls":loss_cls,
            "point_strides" : [8, 16, 32, 64,128],
            "point_base_scale":4,
            "transform_method":'moment',
            "moment_mul":0.01,
            "num_points": cfg.MODEL.REPPOINT_HEAD.NUM_POINTS,
        }
        return ret

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        elif self.transform_method == "exact_minmax":
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_reshape = pts_reshape[:, :2, ...]
            pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
            pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
            bbox_left = pts_x[:, 0:1, ...]
            bbox_right = pts_x[:, 1:2, ...]
            bbox_up = pts_y[:, 0:1, ...]
            bbox_bottom = pts_y[:, 1:2, ...]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    labels, label_weights,
                    bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        # classification loss


        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        #print('len:',len(label_weights))
        #print('sum:',sum(label_weights))
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)


        pos_inds = torch.nonzero(labels != self.num_classes, as_tuple=True)[0]
        oh_labels = torch.zeros_like(cls_score)
        oh_labels[pos_inds, labels[pos_inds]] = 1
        #print('score:',cls_score)
        #print('label:',oh_labels)
        loss_cls = self.loss_cls(cls_score,oh_labels,label_weights,avg_factor=num_total_samples_refine)
        #print("pos_num:",num_total_samples_refine)

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)

        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)


        normalize_term = self.point_base_scale * stride
        pos_init_inds = torch.where(bbox_weights_init > 0.001)[0]
        loss_pts_init = self.loss_bbox_init_weight*self.loss_bbox_init(
            bbox_pred_init[pos_init_inds] / normalize_term,
            bbox_gt_init[pos_init_inds] / normalize_term,
            )/num_total_samples_init
        #pos_inds = torch.where(bbox_weights_refine>0.001)[0]
        #print('num_pos:',len(pos_inds))
        #if len(pos_inds) > 0:
            #show_num = min(len(pos_inds),5)
            #print('pos_pred_bbox:',bbox_pred_refine[pos_inds[:show_num]])
            #print('pos_gt_bbox:',bbox_gt_refine[pos_inds[:show_num]])
        pos_refine_inds = torch.where(bbox_weights_refine>0.001)[0]
        loss_pts_refine = self.loss_bbox_refine_weight*self.loss_bbox_refine(
            bbox_pred_refine[pos_refine_inds] / normalize_term,
            bbox_gt_refine[pos_refine_inds] / normalize_term,
            )/num_total_samples_refine
        return loss_cls, loss_pts_init, loss_pts_refine

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        #print(valid_flag_list[0][0].size())
        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale, scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             num_level_proposals,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             label_channels=1,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 6
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]


        num_level_proposals_inside = self.get_num_level_proposals_inside(num_level_proposals, inside_flags)
        if stage == 'init':
            assigner = self.init_assigner
            assigner_type = "PointMatcher"
            pos_weight = 1.0
        else:
            assigner = self.refine_assigner
            assigner_type = "ATSSAssigner"
            pos_weight = 1.0
        if assigner_type != "ATSSAssigner":
            num_gt, gt_inds, assigned_labels = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, gt_labels)
        else:
            num_gt, gt_inds, assigned_labels = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, gt_bboxes_ignore,
                                            gt_labels)

        pos_inds = torch.nonzero(
            gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals,), self.background_label, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

        pos_assigned_gt_inds = gt_inds[pos_inds] - 1
        if len(pos_inds) > 0:
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
            #print('pos_gt_bbox_0:',pos_gt_bboxes[0])
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_proposals, inside_flags)

        return labels.detach(), label_weights.detach(), bbox_gt.detach(),\
               bbox_weights.detach(), pos_inds.detach(), neg_inds.detach()

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.
        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._point_target_single,
            proposals_list,
            valid_flag_list,
            num_level_proposals_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            stage=stage,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, bbox_weights_list,
                num_total_pos, num_total_neg)

    def forward(self,cls_scores,pts_preds_init,
             pts_preds_refine,targets,pred_only = False):

        if pred_only:
            bbox_preds_refine = []
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            center_list, valid_flag_list = self.get_points(featmap_sizes, targets)


            pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
            for pts_pred_refine in pts_coordinate_preds_refine:
                bbox_pred_refine = self.points2bbox(
                    pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
                bbox_preds_refine.append(bbox_pred_refine)
            return cls_scores,bbox_preds_refine

        gt_bboxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        # TODO: add image_metas

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        #print("ft:{},pt:{}".format( len(featmap_sizes),len(self.point_generators)))
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, targets)

        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)
        # default atss
        candidate_list = center_list
        #if self.train_cfg.init.assigner['type'] != 'MaxIoUAssigner':
            # Assign target for center list

        #else:
            # transform center list to bbox list and
            #   assign target for bbox list
            #bbox_list = self.centers_to_bboxes(center_list)
            #candidate_list = bbox_list

        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            targets,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, targets)

        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())

                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            targets,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)
        (labels_list, label_weights_list,
         bbox_gt_list_refine, bbox_weights_list_refine,
         num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_pos_init,
            num_total_samples_refine=num_total_pos_refine)
        loss_dict_all = {
            'loss_cls': sum(losses_cls),
            'loss_pts_init': sum(losses_pts_init),
            'loss_pts_refine': sum(losses_pts_refine)
        }
        return loss_dict_all
