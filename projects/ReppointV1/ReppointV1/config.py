# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_reppoint_config(cfg):
    cfg.MODEL.REPPOINT_HEAD = CN()
    cfg.MODEL.REPPOINT_HEAD.NAME = "ReppointHead"
    cfg.MODEL.REPPOINT_HEAD.NUM_CLASS = 80
    cfg.MODEL.REPPOINT_HEAD.IN_CHANNEL = 256
    cfg.MODEL.REPPOINT_HEAD.POINT_FEATURE_CHANNEL= 256
    cfg.MODEL.REPPOINT_HEAD.FEATURE_CHANNEL= 256
    cfg.MODEL.REPPOINT_HEAD.STACKED_CONVS= 3
    cfg.MODEL.REPPOINT_HEAD.NUM_POINTS= 9
    cfg.MODEL.REPPOINT_HEAD.GRADIENT_MUL= 0.1
    cfg.MODEL.REPPOINT_HEAD.USE_GRID_POINTS= False
    cfg.MODEL.REPPOINT_HEAD.CENTER_INIT= True
    cfg.MODEL.REPPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5","p6"]


    cfg.MODEL.REPPOINT_LOSS = CN()
    cfg.MODEL.REPPOINT_LOSS.NAME= "ReppointLoss"
    cfg.MODEL.REPPOINT_LOSS.LOSS_CLS_WEIGHT= 1.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_INIT_WEIGHT= 0.5
    cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_REFINE_WEIGHT= 1.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_SMOOTH_BETA= 1.0/9.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_ALPHA= 0.25
    cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_GAMMA= 2.0
    cfg.MODEL.REPPOINT_LOSS.POINT_POS_NUM= 3
    cfg.MODEL.REPPOINT_LOSS.ATSS_TOPK= 9

    cfg.SOLVER.OPTIMIZER = ""
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_reppoint_config(cfg):
    cfg.MODEL.REPPOINT_HEAD = CN()
    cfg.MODEL.REPPOINT_HEAD.NAME = "ReppointHead"
    cfg.MODEL.REPPOINT_HEAD.NUM_CLASS = 80
    cfg.MODEL.REPPOINT_HEAD.IN_CHANNEL = 256
    cfg.MODEL.REPPOINT_HEAD.POINT_FEATURE_CHANNEL= 256
    cfg.MODEL.REPPOINT_HEAD.FEATURE_CHANNEL= 256
    cfg.MODEL.REPPOINT_HEAD.STACKED_CONVS= 3
    cfg.MODEL.REPPOINT_HEAD.NUM_POINTS= 9
    cfg.MODEL.REPPOINT_HEAD.GRADIENT_MUL= 0.1
    cfg.MODEL.REPPOINT_HEAD.USE_GRID_POINTS= False
    cfg.MODEL.REPPOINT_HEAD.CENTER_INIT= True
    cfg.MODEL.REPPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5","p6"]


    cfg.MODEL.REPPOINT_LOSS = CN()
    cfg.MODEL.REPPOINT_LOSS.NAME= "ReppointLoss"
    cfg.MODEL.REPPOINT_LOSS.LOSS_CLS_WEIGHT= 1.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_INIT_WEIGHT= 0.5
    cfg.MODEL.REPPOINT_LOSS.LOSS_BBOX_REFINE_WEIGHT= 1.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_SMOOTH_BETA= 1.0/9.0
    cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_ALPHA= 0.25
    cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_GAMMA= 2.0
    cfg.MODEL.REPPOINT_LOSS.POINT_POS_NUM= 3
    cfg.MODEL.REPPOINT_LOSS.ATSS_TOPK= 9

    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
