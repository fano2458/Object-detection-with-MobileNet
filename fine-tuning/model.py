import torchvision
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class SSDDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ssdlite320_mobilenet_v3_large(torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)

    def forward(self, images, targets=None):
        return self.model(images, targets)


# import torch.nn as nn   
# from torchvision.models import mobilenet_v3_large


# class SSDBoxHead(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.predictor = make_box_predictor(cfg)
#         # 
#         if self.cfg.MODEL.BOX_HEAD.LOSS == 'FocalLoss':
#             self.loss_evaluator = FocalLoss(0.25, 2)
#         else: # By default, we use MultiBoxLoss
#             self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)

#         self.post_processor = PostProcessor(cfg)
#         self.priors = None

#     def forward(self, features, targets=None):
#         cls_logits, bbox_pred = self.predictor(features)
#         if self.training:
#             return self._forward_train(cls_logits, bbox_pred, targets)
#         else:
#             return self._forward_test(cls_logits, bbox_pred)

#     def _forward_train(self, cls_logits, bbox_pred, targets):
#         gt_boxes, gt_labels = targets['boxes'], targets['labels']
#         reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
#         loss_dict = dict(
#             reg_loss=reg_loss,
#             cls_loss=cls_loss,
#         )
#         detections = (cls_logits, bbox_pred)
#         return detections, loss_dict

#     def _forward_test(self, cls_logits, bbox_pred):
#         if self.priors is None:
#             self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
#         # 
#         if self.cfg.MODEL.BOX_HEAD.LOSS == 'FocalLoss':
#             scores = cls_logits.sigmoid()
#         else:
#             scores = F.softmax(cls_logits, dim=2)

#         boxes = box_utils.convert_locations_to_boxes(
#             bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
#         )
        
#         boxes = box_utils.center_form_to_corner_form(boxes)
#         detections = (scores, boxes)
#         detections = self.post_processor(detections)
#         return detections, {}


# class SSDDetector(nn.Module):
#     def __init__(self):
#         self.backbone = 
#         self.box_head = 

#     def forward(self, images, targets=None):
#         features = self.backbone(images)
#         detections, detector_losses = self.box_head(features, targets)
#         if self.training:
#             return detector_losses
#         return detections
    