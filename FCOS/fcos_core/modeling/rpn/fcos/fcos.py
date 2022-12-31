import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
# from .loss import make_fcos_loss_evaluator

from .loss_bin import make_fcos_loss_evaluator
from .loss_bin import make_fcos_loss_evaluator_adv
# from .loss_without_bin import make_fcos_loss_evaluator


from fcos_core.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        bbox_bin_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):  ##4
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

            bbox_bin_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_bin_tower.append(nn.GroupNorm(32, in_channels))
            bbox_bin_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('bbox_bin_tower', nn.Sequential(*bbox_bin_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        num_bin = cfg.BIN.NUM_REG_BIN
        self.bbox_bin_pred = nn.Conv2d(
            in_channels, 4 * num_bin, kernel_size=3, stride=1,
            padding=1
        )
        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.bbox_bin_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness, self.bbox_bin_pred]:

        # for modules in [self.cls_tower, self.bbox_tower,
        #                 self.cls_logits, self.bbox_pred,
        #                 self.centerness, self.bbox_bin_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    ###-------------------------------------------------------------------------------

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        bbox_bin = []

        # adv_logits = []
        # adv_centerness = []
        # adv_bbox_bin = []

        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))

            box_tower = self.bbox_tower(feature)
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(box_tower)
            )))

            bbox_bin.append(self.bbox_bin_pred(self.bbox_bin_tower(feature)))

            ##detach
            # bbox_bin.append(self.bbox_bin_pred(self.bbox_bin_tower(feature.detach())))

            # bbox_bin.append(self.bbox_bin_pred(box_tower))



        # return logits, bbox_reg, centerness, bbox_bin
        return logits, bbox_reg, centerness, bbox_bin


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

        # self.loss_evaluator_adv = make_fcos_loss_evaluator_adv(cfg)

    # def forward(self, images, features, targets=None, return_maps=False):
    def forward(self, images_source, features_source, images_target, features_target, targets=None,
                return_maps=False, box_regression_coarse=None):

        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        if images_target is None and features_target is None:###只在源源域上进行有监督训练或者进行测试
            # box_cls, box_regression, centerness, box_bin, adv_logits, adv_centerness, adv_bbox_bin = self.head(features_source)
            box_cls, box_regression, centerness, box_bin = self.head(
                features_source)

            locations = self.compute_locations(features_source)

            if self.training:
                return self._forward_train(
                    locations, box_cls,
                    box_regression,
                    centerness, box_bin, targets, return_maps, box_regression_coarse
                )
            else:
                return self._forward_test(
                    locations, box_cls, box_regression,
                    centerness, images_source.image_sizes
                )
        else: ###源域和目标域对抗训练
            box_cls_target, box_regression_target, centerness_target, box_bin_target = self.head(features_target)  ###每一项为list
            box_cls_source, box_regression_source, centerness_source, box_bin_source = self.head(features_source)  ###每一项为list


            # locations_source = self.compute_locations(features_source)
            # locations_target = self.compute_locations(features_target)

            return self._forward_train_adv(
                box_cls_source, centerness_source, box_bin_source,
                box_cls_target, centerness_target, box_bin_target,
                targets, return_maps=True
            )



    def _forward_train(self, locations, box_cls, box_regression, centerness, box_bin, targets, return_maps=False, box_regression_coarse=None):
        score_maps = {
            "box_cls": box_cls,
            "box_regression": box_regression,
            "centerness": centerness
        }
        losses = {}
        # print("hehe")
        if targets is not None:
            # print("hehe")
            loss_box_cls, loss_box_reg, loss_centerness, loss_bin = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, box_bin, targets, box_regression_coarse
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness,
                "loss_bin": loss_bin
            }
        else:
            losses = {
                "zero": 0.0 * sum(0.0 * torch.sum(x) for x in box_cls)\
                        +0.0 * sum(0.0 * torch.sum(x) for x in box_regression)\
                        +0.0 * sum(0.0 * torch.sum(x) for x in centerness)\
            }
        if return_maps:
            return None, losses, None, score_maps
        else:
            return None, losses, None

    def _forward_train_adv(self,
            box_cls_source, centerness_source, box_bin_source,
            box_cls_target, centerness_target, box_bin_target,
            targets, return_maps=False):
        score_maps_source = {
            "box_cls": box_cls_source,
            # "box_regression_source": box_regression,
            "centerness": centerness_source
        }
        score_maps_target = {
            "box_cls": box_cls_target,
            # "box_regression_source": box_regression,
            "centerness": centerness_target
        }

        losses = {}
        # losses = {
        #     "zero": 0.0 * sum(0.0 * torch.sum(x) for x in box_cls)\
        #             +0.0 * sum(0.0 * torch.sum(x) for x in box_regression)\
        #             +0.0 * sum(0.0 * torch.sum(x) for x in centerness)\
        # }

        # loss_box_cls, loss_box_reg, loss_centerness, loss_bin = self.loss_evaluator_adv(
        #     box_cls_source, centerness_source, box_bin_source, adv_logits_source, adv_bbox_bin_source,
        #     box_cls_target, centerness_target, box_bin_target, adv_logits_target, adv_bbox_bin_target,
        #     targets
        # )

        if return_maps:
            return None, losses, (box_cls_source, box_cls_target, box_bin_source, box_bin_target), (score_maps_source, score_maps_target)
        else:
            return None, losses, (box_cls_source, box_cls_target, box_bin_source, box_bin_target), None

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}, None

    def compute_locations(self, features): ##特征图上各点反映到原图上的坐标
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
