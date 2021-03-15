from typing import List,Dict
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from functools import partial

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate, DeformConv
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
#from fvcore.nn import sigmoid_focal_loss_jit

REPPOINT_HEAD_REGISTRY = Registry("REPPOINT_HEAD")
REPPOINT_HEAD_REGISTRY.__doc__ = ""


# copy from mmcv
def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if isinstance(module,nn.GroupNorm):
        return 
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments
    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def build_reppoint_heads(cfg):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.REPPOINT_HEAD.NAME
    return REPPOINT_HEAD_REGISTRY.get(name)(cfg)




@REPPOINT_HEAD_REGISTRY.register()
class ReppointHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic described in
    Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self,*,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 num_points=9,
                 gradient_mul=0.1,
                 use_grid_points=False,
                 center_init=True,

                 **kwargs):
        # TODO: losses will set in loss.py
        '''
        Args:
            num_classes: num of pred classes
            in_channels: num of input image channal
            stacked_convs: num of convs used for feature extraction
            feat_channels: num of conv feature
            point_feat_channels: num of dim of points features
            num_points: how much points used to fit an object
            gradient_mul:
            point_strides:
            point_base_scale:
            use_grid_points:
            center_init:
            transform_method:
            moment_mul:
            **kwargs:
        '''
        super(ReppointHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        # [-1.  0.  1.]
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        #[-1. - 1. - 1.  0.  0.  0.  1.  1.  1.]
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        #[-1.  0.  1. -1.  0.  1. -1.  0.  1.]
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        #[-1. - 1. - 1.  0. - 1.  1.  0. - 1.  0.  0.  0.  1.  1. - 1.  1.  0.  1.  1.]
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        # layers
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = []
        self.reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.cls_convs.append(nn.GroupNorm(32,self.feat_channels))
            self.cls_convs.append(nn.ReLU(inplace=True))

            self.reg_convs.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.reg_convs.append(nn.GroupNorm(32,self.feat_channels))
            self.reg_convs.append(nn.ReLU(inplace=True))

        self.cls_convs = nn.Sequential(*self.cls_convs)
        self.reg_convs = nn.Sequential(*self.reg_convs)
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
        # init weight
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

        self.gradient_mul = gradient_mul
        self.cls_out_channels = self.num_classes

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "num_classes":cfg.MODEL.REPPOINT_HEAD.NUM_CLASS,
            "in_channels":cfg.MODEL.REPPOINT_HEAD.IN_CHANNEL,
            "point_feat_channels" : cfg.MODEL.REPPOINT_HEAD.POINT_FEATURE_CHANNEL,
            "feat_channels": cfg.MODEL.REPPOINT_HEAD.FEATURE_CHANNEL,
            "stacked_convs": cfg.MODEL.REPPOINT_HEAD.STACKED_CONVS,
            "num_points": cfg.MODEL.REPPOINT_HEAD.NUM_POINTS,
            "gradient_mul": cfg.MODEL.REPPOINT_HEAD.GRADIENT_MUL,
            "use_grid_points": cfg.MODEL.REPPOINT_HEAD.USE_GRID_POINTS,
            "center_init": cfg.MODEL.REPPOINT_HEAD.CENTER_INIT
        }
        return ret

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

        cls_feat = self.cls_convs(cls_feat)
        pts_feat = self.reg_convs(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))

        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        # TODO: why use grad_mul?
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset

        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))

        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))

        pts_out_refine = pts_out_refine + pts_out_init.detach()
        #cls_out = self.softmax(cls_out)
        #print(pts_out_refine.size())
        #print(pts_out_init.size())
        return cls_out, pts_out_init, pts_out_refine

    def forward(self,x):
        return multi_apply(self.forward_single,x)