# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import Tensor, nn
from torch.nn import functional as F
from .utils import multi_apply,points2bbox,images_to_levels,unmap,offset_to_pts
from .PointsGenerator import generate_all_points
from .assigner import ATSSMatcher,PointMatcher

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from .head import build_reppoint_heads

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

__all__ = ["RepPointV1"]


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class RepPointV1(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        head_in_features,
        init_matcher,
        refine_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.1,
        box_reg_loss_type="giou",
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.6,
        max_detections_per_image=100,
        points_num = 9,
        point_strides = [8, 16, 32, 64, 128],
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        self.points_num = points_num
        self.num_points = points_num
        self.point_strides = point_strides
        self.init_assigner= init_matcher
        self.refine_assigner = refine_matcher

        self.num_classes = num_classes
        self.background_label = num_classes
        self.cls_out_channels = self.num_classes
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format
        self.point_base_scale = 4

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        head = build_reppoint_heads(cfg)
        return {
            "backbone": backbone,
            "head": head,
            "init_matcher":PointMatcher(),
            "refine_matcher":ATSSMatcher(),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.REPPOINT_HEAD.NUM_CLASS,
            "head_in_features": cfg.MODEL.REPPOINT_HEAD.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_ALPHA,
            "focal_loss_gamma": cfg.MODEL.REPPOINT_LOSS.LOSS_FOCAL_GAMMA,
            "smooth_l1_beta": cfg.MODEL.REPPOINT_LOSS.LOSS_SMOOTH_BETA,
            "box_reg_loss_type": cfg.MODEL.REPPOINT_LOSS.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.REPPOINT_LOSS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.REPPOINT_LOSS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.REPPOINT_LOSS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        images,img_metas = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        featmap_sizes = [feature.size()[-2:] for feature in features]

        pred_logits, pts_pred_init, pts_pred_refine = self.head(features)

        center_list, valid_flag_list = generate_all_points(featmap_sizes, img_metas)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_bboxes = [gt_instance.gt_boxes.tensor for gt_instance in gt_instances]
            gt_labels = [gt_instance.gt_classes for gt_instance in gt_instances]
            pts_coordinate_preds_init = offset_to_pts(center_list, pts_pred_init,
                                          self.point_strides, self.points_num)
            candidate_list = center_list
            cls_reg_targets_init = self.get_targets(
                candidate_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_labels_list=gt_labels,
                stage='init',
                label_channels=1)
            (*_, bbox_gt_list_init, bbox_weights_list_init,
             num_total_pos_init, num_total_neg_init) = cls_reg_targets_init

            center_list, valid_flag_list = generate_all_points(featmap_sizes, img_metas)
            pts_coordinate_preds_refine = offset_to_pts(center_list, pts_pred_refine)
            bbox_list = []
            for i_img, center in enumerate(center_list):
                bbox = []
                for i_lvl in range(len(pts_pred_refine)):
                    bbox_preds_init = points2bbox(
                        pts_pred_init[i_lvl].detach())
                    bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                    bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                    bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                bbox_list.append(bbox)
            #print('bbox_list:',len(bbox_list))
            cls_reg_targets_refine = self.get_targets(
                bbox_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=None,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=1)
            (labels_list, label_weights_list,
             bbox_gt_list_refine, bbox_weights_list_refine,
             num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

            losses = self.losses(
                pred_logits,
                pts_coordinate_preds_init,
                pts_coordinate_preds_refine,
                labels_list,
                label_weights_list,
                bbox_gt_list_init,
                bbox_weights_list_init,
                bbox_gt_list_refine,
                bbox_weights_list_refine,
                num_total_pos_init,
                num_total_pos_refine)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(pred_logits, pts_pred_refine,
                         center_list, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred_logits, pts_pred_refine,
                                     center_list, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    labels, label_weights,
                    bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        valid_mask = torch.where(label_weights>0.01)[0]
        gt_labels_target = F.one_hot(labels[valid_mask], num_classes=self.num_classes + 1)[
                           :, :-1
                           ]
        loss_cls = sigmoid_focal_loss_jit(
            cls_score[valid_mask],
            gt_labels_target.to(cls_score[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )/num_total_samples_refine

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride


        if self.box_reg_loss_type == "smooth_l1":
            pos_mask = torch.where(bbox_weights_init > 0.001)[0]
            loss_pts_init = 0.5*smooth_l1_loss(
                bbox_pred_init[pos_mask]/normalize_term,
                bbox_gt_init[pos_mask]/normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )/num_total_samples_init/4.0

            pos_mask = torch.where(bbox_weights_refine > 0.001)[0]
            loss_pts_refine = smooth_l1_loss(
                bbox_pred_refine[pos_mask]/normalize_term,
                bbox_gt_refine[pos_mask]/normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )/num_total_samples_refine/4.0

        elif self.box_reg_loss_type == "giou":
            pos_mask = torch.where(bbox_weights_init > 0.001)[0]
            loss_pts_init = 0.5*giou_loss(
                bbox_pred_init[pos_mask] / normalize_term,
                bbox_gt_init[pos_mask] / normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / num_total_samples_init/4.0

            pos_mask = torch.where(bbox_weights_refine > 0.001)[0]
            loss_pts_refine = giou_loss(
                bbox_pred_refine[pos_mask] / normalize_term,
                bbox_gt_refine[pos_mask] / normalize_term,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / num_total_samples_refine/4.0
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_cls, loss_pts_init, loss_pts_refine

    def losses(self, cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
               num_total_pos_init,
               num_total_pos_refine
               ):
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
    @torch.no_grad()
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

        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        #TODO:
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
            num_gt, gt_inds, assigned_labels = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes,
                                                               gt_bboxes_ignore, gt_labels)

        pos_inds = torch.nonzero(
            gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        #print('neg:',len(neg_inds))


        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals,), self.background_label, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_assigned_gt_inds = gt_inds[pos_inds] - 1

        if len(pos_inds) > 0:
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
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


    @torch.no_grad()
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
        res = multi_apply(
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
        #print('res:',len(res))
        (all_labels, all_label_weights, all_bbox_gt, all_bbox_weights,
         pos_inds_list, neg_inds_list) = res
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

    def inference(
        self,
        pred_logits: List[Tensor],
        pts_pred_refine: List[Tensor],
        center_lists: List[Tensor],
        image_sizes: List[Tuple[int,int]]
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        pts_coordinate_preds_refine = offset_to_pts(center_lists, pts_pred_refine)
        pred_bboxes = [
            points2bbox(x.reshape(-1,2*self.points_num)).view(x.size()[0],-1,4) for x in pts_coordinate_preds_refine]
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            pred_bboxes_per_image = \
                [x[img_idx] for x in pred_bboxes]
            results_per_image = self.inference_single_image(
                pred_logits_per_image, pred_bboxes_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        pred_logits: List[Tensor],
        pred_bboxes: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i in zip(pred_logits, pred_bboxes):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold

            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only

            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            predicted_boxes = box_reg_i[anchor_idxs]

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images_pad_hw = [{'pad_shape':x["image"].to(self.device).size()[-2:]} for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images,images_pad_hw


