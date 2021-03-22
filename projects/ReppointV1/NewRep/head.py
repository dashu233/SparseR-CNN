import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from detectron2.config import configurable
from detectron2.layers.deform_conv import DeformConv
from detectron2.layers.wrappers import Conv2d
from detectron2.layers.batch_norm import get_norm
from detectron2.layers.nms import batched_nms

from detectron2.utils.registry import Registry
from functools import partial
from .assigner import *

REPPOINT_HEAD_REGISTRY = Registry("REPPOINT_HEAD")
REPPOINT_HEAD_REGISTRY.__doc__ = ""


from .mmcv_utils import *


def build_reppoint_heads(cfg):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.REPPOINT_HEAD.NAME
    return REPPOINT_HEAD_REGISTRY.get(name)(cfg)


# TODO:cfg,result

@REPPOINT_HEAD_REGISTRY.register()
class RepPointsHead(nn.Module):
    """RepPoint head.
    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605
    @configurable()
    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 stacked_convs = 4,
                 point_base_scale=4,
                 loss_cls_gamma=2.0,
                 loss_cls_alpha=0.25,
                 loss_cls_weight = 1.0,
                 box_reg_loss_type = "smooth_l1",
                 smooth_beta = 1.0/9.0,
                 pts_refine_loss_weight = 1.0,
                 pts_init_loss_weight = 0.5,
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01,
                 test_score_thresh=0.05,
                 test_topk_candidates=1000,
                 test_nms_thresh = 0.6,
                 max_detections_per_image=100,
                 **kwargs):
        super(RepPointsHead, self).__init__()
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.focal_loss_alpha = loss_cls_alpha
        self.focal_loss_gamma = loss_cls_gamma
        self.loss_cls_weight = loss_cls_weight

        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_beta
        self.pts_refine_loss_weight = pts_refine_loss_weight
        self.pts_init_loss_weight = pts_init_loss_weight
        self.test_score_thresh=test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs

        self.fp16_enabled = False
        self.background_label = num_classes
        self.norm = "GN"


        self.init_assigner = PointMatcher()
        self.refine_assigner = ATSSMatcher()

        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        self.cls_out_channels = self.num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            #print(self.feat_channels)
            self.cls_convs.append(
                Conv2d(
                    in_channels=chn,
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=get_norm(self.norm,self.feat_channels),
                    activation=self.relu
                ))
            #print('2:',self.feat_channels)
            self.reg_convs.append(
                Conv2d(
                    in_channels=chn,
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=get_norm(self.norm, self.feat_channels),
                    activation=self.relu
                ))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.REPPOINT_HEAD.NUM_CLASSES,
            "in_channels": cfg.MODEL.REPPOINT_HEAD.IN_CHANNEL,
            "point_feat_channels": cfg.MODEL.REPPOINT_HEAD.POINT_FEATURE_CHANNEL,
            "feat_channels": cfg.MODEL.REPPOINT_HEAD.FEATURE_CHANNEL,
            "stacked_convs": cfg.MODEL.REPPOINT_HEAD.STACKED_CONVS,
            "num_points": cfg.MODEL.REPPOINT_HEAD.NUM_POINTS,
            "gradient_mul": cfg.MODEL.REPPOINT_HEAD.GRADIENT_MUL,
            "use_grid_points": cfg.MODEL.REPPOINT_HEAD.USE_GRID_POINTS,
            "center_init": cfg.MODEL.REPPOINT_HEAD.CENTER_INIT,
            "loss_cls_weight" : cfg.MODEL.REPPOINT_LOSS.LOSS_CLS_WEIGHT,
            "pts_refine_loss_weight" : cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_REFINE_WEIGHT,
            "pts_init_loss_weight" : cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_INIT_WEIGHT,
            "transform_method" : cfg.MODEL.REPPOINT_HEAD.TRANSFORM_METHOD,
            "point_strides" : cfg.MODEL.REPPOINT_HEAD.POINT_STRIDES,
            "point_base_scale" : cfg.MODEL.REPPOINT_HEAD.POINT_BAST_SCALE,
            # Loss parameters:
            "loss_cls_alpha": cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_ALPHA,
            "loss_cls_gamma": cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_GAMMA,
            "smooth_beta": cfg.MODEL.REPPOINT_LOSS.LOSS_SMOOTH_BETA,
            "box_reg_loss_type": cfg.MODEL.REPPOINT_LOSS.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.REPPOINT_LOSS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.REPPOINT_LOSS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.REPPOINT_LOSS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
            normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

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

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """ Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine

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

        return points_list, valid_flag_list

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
            return (None, ) * 6
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        num_level_proposals_inside = self.get_num_level_proposals_inside(num_level_proposals, inside_flags)
        if stage == 'init':
            assigner = self.init_assigner
            assigner_type = ""
            pos_weight = 1
        else:
            assigner = self.refine_assigner
            assigner_type = "ATSSAssigner"
            pos_weight = 1
        if assigner_type != "ATSSAssigner":
            num_gt, gt_inds, assigned_labels = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, gt_labels)
        else:
            num_gt, gt_inds, assigned_labels = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)

        pos_inds = torch.nonzero(
            gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ), self.background_label, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_assigned_gt_inds = gt_inds[pos_inds] - 1

        if len(pos_inds) > 0:
            pos_gt_bboxes =  gt_bboxes[pos_assigned_gt_inds]
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

        return labels, label_weights, bbox_gt, bbox_weights, pos_inds, neg_inds

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

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    labels, label_weights,
                    bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        pos_labels_inds = torch.where(label_weights)[0]
        gt_labels_target = F.one_hot(labels[pos_labels_inds], num_classes=self.num_classes + 1)[
                           :, :-1
                           ]
        loss_cls = sigmoid_focal_loss_jit(
            cls_score[pos_labels_inds],
            gt_labels_target.to(cls_score.dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )/num_total_samples_refine*self.loss_cls_weight

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
        if self.box_reg_loss_type == "smooth_l1":
            pos_mask = torch.where(bbox_weights_init > 0)[0]
            loss_pts_init = smooth_l1_loss(
                bbox_pred_init[pos_mask] / normalize_term,
                bbox_gt_init[pos_mask] / normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum"
            ) / num_total_samples_init * self.pts_init_loss_weight/4.0

            pos_mask = torch.where(bbox_weights_refine > 0)[0]
            loss_pts_refine = smooth_l1_loss(
                bbox_pred_refine[pos_mask] / normalize_term,
                bbox_gt_refine[pos_mask] / normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum"
            ) / num_total_samples_refine * self.pts_refine_loss_weight/4.0

        elif self.box_reg_loss_type == "giou":
            pos_mask = torch.where(bbox_weights_init>0)[0]
            loss_pts_init = giou_loss(
               bbox_pred_init[pos_mask]/normalize_term,
                bbox_gt_init[pos_mask]/normalize_term,
                reduction="sum"
            )/num_total_samples_init*self.pts_init_loss_weight

            pos_mask = torch.where(bbox_weights_refine > 0)[0]
            loss_pts_refine = giou_loss(
                bbox_pred_refine[pos_mask] / normalize_term,
                bbox_gt_refine[pos_mask] / normalize_term,
                reduction="sum"
            ) / num_total_samples_refine * self.pts_refine_loss_weight

        return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)

        candidate_list = center_list

        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
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
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
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

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   img_metas,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [self.points2bbox(pts_pred_refine) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_points, img_shape,
                                                nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(zip(cls_scores, bbox_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = self.test_topk_candidates
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        max_scores,labels = torch.max(mlvl_scores,dim=1)
        if nms:
            keep = batched_nms(mlvl_bboxes, max_scores.view(-1),labels.view(-1),self.test_score_thresh)
            keep = keep[: self.max_detections_per_image]
            return mlvl_bboxes[keep],labels[keep],max_scores[keep]
        else:
            return mlvl_bboxes, labels

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside