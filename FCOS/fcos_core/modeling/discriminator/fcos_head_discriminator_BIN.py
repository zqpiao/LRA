import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


class FCOSDiscriminator_BIN(nn.Module):
    def __init__(self, cfg, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, center_aware_type='ca_loss', grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_BIN, self).__init__()

        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        ###--------------------------定义对抗判别器----------------------
        adv_cls_tower = []
        adv_bbox_tower = []
        adv_bbox_bin_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):  ##4
            # adv_cls_tower.append(
            #     nn.Conv2d(
            #         in_channels,
            #         in_channels,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1
            #     )
            # )
            # adv_cls_tower.append(nn.GroupNorm(32, in_channels))
            # adv_cls_tower.append(nn.ReLU())
            adv_bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            adv_bbox_tower.append(nn.GroupNorm(32, in_channels))
            adv_bbox_tower.append(nn.ReLU())

            # adv_bbox_bin_tower.append(
            #     nn.Conv2d(
            #         in_channels,
            #         in_channels,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1
            #     )
            # )
            # adv_bbox_bin_tower.append(nn.GroupNorm(32, in_channels))
            # adv_bbox_bin_tower.append(nn.ReLU())

        self.add_module('adv_cls_tower', nn.Sequential(*adv_cls_tower))
        self.add_module('adv_bbox_tower', nn.Sequential(*adv_bbox_tower))
        self.add_module('adv_bbox_bin_tower', nn.Sequential(*adv_bbox_bin_tower))
        # self.adv_cls_logits = nn.Conv2d(
        #     in_channels, num_classes, kernel_size=3, stride=1,
        #     padding=1
        # )
        # self.adv_cls_logits_new = nn.Conv2d(
        #     in_channels, num_classes, kernel_size=3, stride=1,
        #     padding=1
        # )
        # self.adv_centerness = nn.Conv2d(
        #     in_channels, 1, kernel_size=3, stride=1,
        #     padding=1
        # )
        num_bin = cfg.BIN.NUM_REG_BIN
        self.adv_bbox_bin_pred = nn.Conv2d(
            in_channels, 4 * num_bin, kernel_size=3, stride=1,
            padding=1
        )

        # self.grad_reverse_lambda = cfg.MODEL.ADV.GRL_WEIGHT_P3
        self.grad_reverse = GradientReversal(grad_reverse_lambda)

        # initialization
        for modules in [self.adv_bbox_tower,
                        self.adv_bbox_bin_pred]:

            # for modules in [self.cls_tower, self.bbox_tower,
            #                 self.cls_logits, self.bbox_pred,
            #                 self.centerness, self.bbox_bin_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # # hyperparameters
        # assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
        # self.center_aware_weight = center_aware_weight
        # self.center_aware_type = center_aware_type
        #
        # assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        # self.grl_applied_domain = grl_applied_domain
        self.ca_dis_lambda = cfg.MODEL.ADV.CA_DIS_LAMBDA
        self.ga_dis_lambda = cfg.MODEL.ADV.GA_DIS_LAMBDA
        self.use_dis_global = cfg.MODEL.ADV.USE_DIS_GLOBAL
        self.use_dis_ca = cfg.MODEL.ADV.USE_DIS_CENTER_AWARE

        self.center_aware_weight = center_aware_weight


    def forward(self, box_bin_source, box_bin_target,
                features_s, features_t, score_maps_s, score_maps_t):
        # assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        # assert domain == 'source' or domain == 'target'

        ###对抗器输出
        ##----------------------对抗判别器------------------------
        feature_s = self.grad_reverse(features_s)

        # adv_centerness.append(self.adv_centerness(adv_cls_tower))
        adv_box_tower_s = self.adv_bbox_tower(feature_s)
        adv_bbox_bin_source = self.adv_bbox_bin_pred(adv_box_tower_s)


        feature_t = self.grad_reverse(features_t)
                # adv_centerness.append(self.adv_centerness(adv_cls_tower))
        adv_box_tower_t = self.adv_bbox_tower(feature_t)
        # bbox_bin.append(self.bbox_bin_pred(self.bbox_bin_tower(feature)))
        adv_bbox_bin_target = self.adv_bbox_bin_pred(adv_box_tower_t)
        ##--------------------------------------------------------

        # Generate cneter-aware map
        box_cls_map_s = score_maps_s["box_cls"].clone().sigmoid()
        centerness_map_s = score_maps_s["centerness"].clone().sigmoid()
        n, c, h, w = box_cls_map_s.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map_s = maxpooling(box_cls_map_s)
        # Normalize the center-aware map
        atten_map_s = (self.center_aware_weight * box_cls_map_s * centerness_map_s).sigmoid()

        box_cls_map_t = score_maps_t["box_cls"].clone().sigmoid()
        centerness_map_t = score_maps_t["centerness"].clone().sigmoid()
        n, c, h, w = box_cls_map_t.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map_t = maxpooling(box_cls_map_t)
        # Normalize the center-aware map
        atten_map_t = (self.center_aware_weight * box_cls_map_t * centerness_map_t).sigmoid()

        ##------------------对抗判别损失--------------------
        loss_adv_box_bin = 0
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ####box_bin:      ++++++++++++++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ###----------------attention Map---------------------
        # Generate cneter-aware map
        ###源域
        # loss_adv_box_bin_l, loss_adv_box_bin_t, loss_adv_box_bin_b, loss_adv_box_bin_r = 0,0,0,0
        ###预测器中l,t,b,r分别处理
        box_bin_source_level = box_bin_source  ##batch, 4*bin,h,w
        num_reg_bin = int(box_bin_source_level.size(1) / 4)
        box_bin_source_level_l, box_bin_source_level_t, box_bin_source_level_b, box_bin_source_level_r \
            = box_bin_source_level[:, :num_reg_bin, :, :], box_bin_source_level[:, num_reg_bin:2 * num_reg_bin, :,
                                                           :], box_bin_source_level[:,
                                                               2 * num_reg_bin:3 * num_reg_bin, :, :] \
            , box_bin_source_level[:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

        # box_bin_source_level_l, box_bin_source_level_t, box_bin_source_level_b, box_bin_source_level_r =\
        #     F.softmax(box_bin_source_level_l,dim=1), F.softmax(box_bin_source_level_t,dim=1), \
        #     F.softmax(box_bin_source_level_b,dim=1), F.softmax(box_bin_source_level_r,dim=1)

        ###对抗预测器adv中l,t,b,r分别处理
        adv_box_bin_source_level_l, adv_box_bin_source_level_t, adv_box_bin_source_level_b, adv_box_bin_source_level_r \
            = adv_bbox_bin_source[:, :num_reg_bin, :, :], adv_bbox_bin_source[:, num_reg_bin:2 * num_reg_bin,
                                                             :, :], \
              adv_bbox_bin_source[:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
            , adv_bbox_bin_source[:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

        box_bin_source_level_l_argmax, box_bin_source_level_t_argmax, box_bin_source_level_b_argmax, box_bin_source_level_r_argmax \
            = box_bin_source_level_l.max(1)[1], box_bin_source_level_t.max(1)[1], box_bin_source_level_b.max(1)[1], \
              box_bin_source_level_r.max(1)[1]

        ###目标域
        ###预测器中l,t,b,r分别处理
        box_bin_target_level = box_bin_target  ##batch, 4*bin,h,w
        box_bin_target_level_l, box_bin_target_level_t, box_bin_target_level_b, box_bin_target_level_r \
            = box_bin_target_level[:, :num_reg_bin, :, :], box_bin_target_level[:, num_reg_bin:2 * num_reg_bin, :,
                                                           :], \
              box_bin_target_level[:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
            , box_bin_target_level[:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

        # box_bin_target_level_l, box_bin_target_level_t, box_bin_target_level_b, box_bin_target_level_r = \
        #     F.softmax(box_bin_target_level_l, dim=1), F.softmax(box_bin_target_level_t, dim=1), \
        #     F.softmax(box_bin_target_level_b, dim=1), F.softmax(box_bin_target_level_r, dim=1)

        ###对抗预测器adv中l,t,b,r分别处理
        adv_box_bin_target_level_l, adv_box_bin_target_level_t, adv_box_bin_target_level_b, adv_box_bin_target_level_r \
            = adv_bbox_bin_target[:, :num_reg_bin, :, :], adv_bbox_bin_target[:, num_reg_bin:2 * num_reg_bin,
                                                             :, :], \
              adv_bbox_bin_target[:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
            , adv_bbox_bin_target[:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

        box_bin_target_level_l_argmax, box_bin_target_level_t_argmax, box_bin_target_level_b_argmax, box_bin_target_level_r_argmax \
            = box_bin_target_level_l.max(1)[1], box_bin_target_level_t.max(1)[1], box_bin_target_level_b.max(1)[1], \
              box_bin_target_level_r.max(1)[1]

        # adv_box_bin_l_log_source = torch.log(_sigmoid(adv_bbox_bin_source[l]))
        # adv_box_bin_l_log_source = adv_box_bin_l_log_source * atten_map_source
        # loss_adv_box_bin_source = F.nll_loss(adv_box_bin_l_log_source, box_bin_source_argmax)

        # adv_box_bin_t_log_source = torch.log(F.softmax(adv_box_bin_source_level_t, dim=1))
        # adv_box_bin_t_log_source = adv_box_bin_t_log_source * atten_map_source
        # loss_adv_box_bin_source_t = F.nll_loss(adv_box_bin_t_log_source, box_bin_source_level_t_argmax)

        eps = 1e-5

        adv_box_bin_l_log_source = torch.log(adv_box_bin_source_level_l.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_l_log_source = adv_box_bin_l_log_source * atten_map_s
        loss_adv_box_bin_source_l = F.nll_loss(adv_box_bin_l_log_source, box_bin_source_level_l_argmax)

        adv_box_bin_l_log_target = torch.log(1 - adv_box_bin_target_level_l.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_l_log_target = adv_box_bin_l_log_target * atten_map_t
        loss_adv_box_bin_target_l = F.nll_loss(adv_box_bin_l_log_target, box_bin_target_level_l_argmax)

        loss_adv_box_bin_l = loss_adv_box_bin_source_l + loss_adv_box_bin_target_l
        ##--------------------t
        adv_box_bin_t_log_source = torch.log(adv_box_bin_source_level_t.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_t_log_source = adv_box_bin_t_log_source * atten_map_s
        loss_adv_box_bin_source_t = F.nll_loss(adv_box_bin_t_log_source, box_bin_source_level_t_argmax)

        adv_box_bin_t_log_target = torch.log(1 - adv_box_bin_target_level_t.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_t_log_target = adv_box_bin_t_log_target * atten_map_t
        loss_adv_box_bin_target_t = F.nll_loss(adv_box_bin_t_log_target, box_bin_target_level_t_argmax)

        loss_adv_box_bin_t = loss_adv_box_bin_source_t + loss_adv_box_bin_target_t
        ##--------------------b
        adv_box_bin_b_log_source = torch.log(adv_box_bin_source_level_b.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_b_log_source = adv_box_bin_b_log_source * atten_map_s
        loss_adv_box_bin_source_b = F.nll_loss(adv_box_bin_b_log_source, box_bin_source_level_b_argmax)

        adv_box_bin_b_log_target = torch.log(1 - adv_box_bin_target_level_b.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_b_log_target = adv_box_bin_b_log_target * atten_map_t
        loss_adv_box_bin_target_b = F.nll_loss(adv_box_bin_b_log_target, box_bin_target_level_b_argmax)

        loss_adv_box_bin_b = loss_adv_box_bin_source_b + loss_adv_box_bin_target_b
        ##--------------------r
        adv_box_bin_r_log_source = torch.log(adv_box_bin_source_level_r.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_r_log_source = adv_box_bin_r_log_source * atten_map_s
        loss_adv_box_bin_source_r = F.nll_loss(adv_box_bin_r_log_source, box_bin_source_level_r_argmax)

        adv_box_bin_r_log_target = torch.log(1 - adv_box_bin_target_level_r.sigmoid() + eps)
        if self.use_dis_ca:
            adv_box_bin_r_log_target = adv_box_bin_r_log_target * atten_map_t
        loss_adv_box_bin_target_r = F.nll_loss(adv_box_bin_r_log_target, box_bin_target_level_r_argmax)

        loss_adv_box_bin_r = loss_adv_box_bin_source_r + loss_adv_box_bin_target_r
        ##-----------------------------------------------------------------------------------
        ###city2fog
        loss_adv_box_bin = (loss_adv_box_bin_l + loss_adv_box_bin_t + loss_adv_box_bin_b + loss_adv_box_bin_r) / 4
        ###kitti
        # loss_adv_box_bin = loss_adv_box_bin_l + loss_adv_box_bin_t + loss_adv_box_bin_b + loss_adv_box_bin_r

        # loss_adv_box_bin = loss_adv_box_bin_ltbr

        # num_layers = len(box_cls_source)
        # loss_adv_box_cls, loss_adv_centerness, loss_adv_box_bin = loss_adv_box_cls / num_layers, loss_adv_centerness  / num_layers, loss_adv_box_bin  / num_layers
        # return loss_adv_box_cls, loss_adv_centerness, loss_adv_box_bin

        # loss_adv_box_cls, loss_adv_box_bin = loss_adv_box_cls / num_layers, loss_adv_box_bin / num_layers

        # loss_adv_box_cls, loss_adv_box_bin = loss_adv_box_cls, loss_adv_box_bin

        # if self.use_dis_ca:  ##with CA
        #     loss_adv_box_cls, loss_adv_box_bin = self.ca_dis_lambda * loss_adv_box_cls, self.ca_dis_lambda * loss_adv_box_bin
        # else:  ##GA only
        #     loss_adv_box_cls, loss_adv_box_bin = self.ga_dis_lambda * loss_adv_box_cls, self.ga_dis_lambda * loss_adv_box_bin

        # cls_bin_gama = 0.7   ###city2fog
        # cls_bin_gama = 0.1  ##city2fog 43.4
        cls_bin_gama = 0.2  ##sim  kitti

        # cls_bin_gama = 0.1  ##sim  ResNet101

        return cls_bin_gama * loss_adv_box_bin
        # return cls_bin_gama * loss_adv_box_cls + (1 - cls_bin_gama) * loss_adv_box_bin

        # return loss_adv_box_cls + loss_adv_box_bin
