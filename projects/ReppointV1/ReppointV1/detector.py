import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from detectron2.layers import batched_nms
from torch import nn
from .fpn6 import build_resnet_fpnp6_backbone

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY,Backbone, build_backbone,\
    detector_postprocess
from .head import build_reppoint_heads
from .loss import build_reppoint_loss

from detectron2.structures import Boxes, ImageList, Instances

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n



@META_ARCH_REGISTRY.register()
class ReppointV1(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            reppoint_heads: nn.Module,
            criterion: nn.Module,
            num_classes: int,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            in_features:List[str]=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.points_heads = reppoint_heads
        self.in_features = in_features
        self.criterion = criterion
        self.input_format = input_format
        self.num_classes = num_classes
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        reppoint_heads = build_reppoint_heads(cfg)
        criterion = build_reppoint_loss(cfg)
        return {
            "backbone": backbone,
            "reppoint_heads": reppoint_heads,
            "criterion": criterion,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_format": None, # used for visualization
            "in_features":cfg.MODEL.REPPOINT_HEAD.IN_FEATURES,
            "num_classes":cfg.MODEL.REPPOINT_HEAD.NUM_CLASS
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."

        images,images_pad_hw = self.preprocess_image(batched_inputs)
        src = self.backbone(images.tensor)
        # for COCO's original picture, output is a 7*7 feature map
        # when use different stage, the output shape may change

        features = list()
        # combine different stage feature
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        #print(self.in_features)
        #print(src.keys())
        cls_out, pts_out_init, pts_out_refine = self.points_heads(features)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances,images_pad_hw)
        loss_dict = self.criterion(cls_out, pts_out_init, pts_out_refine, targets)

        #print(loss_dict)
        return loss_dict

    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = False,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        images, images_pad_hw = self.preprocess_image(batched_inputs)
        src = self.backbone(images.tensor)
        targets = [{'pad_shape':hw} for hw in images_pad_hw]
        # for COCO's original picture, output is a 7*7 feature map
        # when use different stage, the output shape may change

        features = list()
        # combine different stage feature
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        cls_out, pts_out_init, pts_out_refine = self.points_heads(features)
        cls_scores,bbox_preds_refine = self.criterion(cls_out, pts_out_init, pts_out_refine, targets,True)

        batch_size = len(images)
        for i,(cs,bbox) in enumerate(zip(cls_scores,bbox_preds_refine)):
            cls_scores[i] = cs.view(batch_size,80,-1).permute(2,0,1)
            bbox_preds_refine[i] = bbox.view(batch_size,4,-1).permute(2,0,1)

            #print(cls_scores[i].size())
        results = []
        cls_score = torch.cat(cls_scores,dim=0).view(batch_size,-1,self.num_classes)
        labels = torch.argmax(cls_score,dim=2)
        scores = torch.max(cls_score,dim=2)
        bbox_pred_refine = torch.cat(bbox_preds_refine,dim=0).view(batch_size,-1,4)
        for i,(score_i,label_i,bbox_pred),in enumerate(zip(scores,labels,bbox_pred_refine)):
            result = Instances(targets[i]['pad_shape'])
            score_i = score_i.view(-1)
            label_i = label_i.view(-1)
            norm_bbox = False
            if norm_bbox:
                h,w = images_pad_hw[i]
                whwh = torch.as_tensor([w,h,w,h],device=scores.device)
                bbox_pred = bbox_pred.view(-1,4)*whwh
            else:
                bbox_pred = bbox_pred.view(-1, 4)
            pos_id = torch.where(label_i < self.num_classes)[0]
            bbox_pred = bbox_pred[pos_id,:]
            score_i = score_i[pos_id]
            label_i = label_i[pos_id]

            idx = batched_nms(bbox_pred,score_i,label_i,0.5)
            #print(idx)

            result.pred_boxes = Boxes(bbox_pred[idx])
            result.scores = score_i[idx]
            result.pred_classes = label_i[idx]
            results.append(result)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})

        return processed_results


    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_pad_hw = list()
        for ts in images.tensor:
            hw = ts.shape[-2:]
            images_pad_hw.append(np.array([hw[0],hw[1]]))
        #print('pad_wh_shape')
        #print(images_pad_whwh)

        return images, images_pad_hw

    def prepare_targets(self, targets,image_pad_hw):
        new_targets = []
        for i,targets_per_image in enumerate(targets):
            target = {}
            h,w = image_pad_hw[i]
            gt_classes = targets_per_image.gt_classes
            norm_bbox = False
            if norm_bbox:
                image_pad_whwh = torch.as_tensor([w, h, w, h],device=targets_per_image.gt_boxes.tensor.device)

                gt_boxes = targets_per_image.gt_boxes.tensor/image_pad_whwh
            else:
                gt_boxes = targets_per_image.gt_boxes.tensor
            #print('gt_boxes:',gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            # unnormalized xyxy

            target["boxes"] = gt_boxes.to(self.device)
            target["pad_shape"] = [h,w]
            new_targets.append(target)

        return new_targets

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results