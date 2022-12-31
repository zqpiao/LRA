"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist

import os
import numpy as np


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

INF = 100000000

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg, iou_th=0.7):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_bin = cfg.BIN.NUM_REG_BIN
        self.center_aware_weight = cfg.MODEL.ADV.CENTER_AWARE_WEIGHT
        self.class_criterion = nn.CrossEntropyLoss()

        self.iou_th = iou_th

    def prepare_targets(self, points, targets, num_bin):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)##(20604,2)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, reg_bin_targets, bin_range = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, num_bin
        )  ###所有层上点的坐标揉在一块  20604个（15456+3864+966+252+66）

        for i in range(len(labels)): ###per batch
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            reg_bin_targets[i] = torch.split(reg_bin_targets[i], num_points_per_level, dim=0)

            bin_range[i] = torch.split(bin_range[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        reg_bin_targets_level_first = []
        bin_range_level_first = []

        for level in range(len(points)):  ###5 feature maps num
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )  ###将各batch按feature map cat起来
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )
            reg_bin_targets_level_first.append(
                torch.cat([reg_bin_targets_per_im[level] for reg_bin_targets_per_im in reg_bin_targets], dim=0)
            )
            bin_range_level_first.append(
                torch.cat([bin_range_per_im[level] for bin_range_per_im in bin_range], dim=0)
            )
        ###labels_level_first(92376)(23184)(5796)(1512)(396)  92376 =  15456 * 6
        ###reg_targets_level_first(92376,4)(23184,4)(5796,4)(1512,4)(396,4)
        # print('labels_level_first.size: ', labels_level_first[0].size())
        # print('reg_targets_level_first.size: ', reg_targets_level_first[0].size())
        return labels_level_first, reg_targets_level_first, reg_bin_targets_level_first, bin_range_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest,num_bin):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        reg_bin_targets = []  ##（20604,4)  not one-hot
        # num_bin = 10
        bin_range = object_sizes_of_interest[:, [1]]    ##(20604,1)
        # bin_range = object_sizes_of_interest[:, 0]  ##(20604)
        bin_range[bin_range == INF] = 1024

        bin_range_batch = []

        for im_i in range(len(targets)):  ##per img in a batch
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox  ###(n,4)  n个bbox
            labels_per_im = targets_per_im.get_field("labels")  ##从1开始
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]  ##(20604,bbox_num_per_img)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)  ##(20604, bbox_num_per_img, 4)注意是stack不是cat

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF  ##(20604,bbox_num_per_img)

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds] ##(20604,4) #之前是stack
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)   ##list
            reg_targets.append(reg_targets_per_im)

            ###-------------对l,t,r,b的bin进行编码-------------

            # offset =  bin_range // 4
            # l, t, b, r = l+offset, t+offset, b+offset, r+offset


            # delt_bin = bin_range  / 2 / num_bin  ##the width of each bin
            delt_bin = bin_range / num_bin

            l_bin_label = (l // delt_bin).int() + 1  ###the label of the first bin is 1
            t_bin_label = (t // delt_bin).int() + 1
            b_bin_label = (b // delt_bin).int() + 1
            r_bin_label = (r // delt_bin).int() + 1

            ###超过512+512的均放在最后一个bin
            l_bin_label[l_bin_label > num_bin] = num_bin
            t_bin_label[t_bin_label > num_bin] = num_bin
            b_bin_label[b_bin_label > num_bin] = num_bin
            r_bin_label[r_bin_label > num_bin] = num_bin

            l_bin_label[l_bin_label < 1] = 1
            t_bin_label[t_bin_label < 1] = 1
            b_bin_label[b_bin_label < 1] = 1
            r_bin_label[r_bin_label < 1] = 1

            reg_bin_targets_per_im = torch.stack([l_bin_label, t_bin_label, b_bin_label, r_bin_label], dim=2) ##(20604,bbox_num_per_img,4)

            reg_bin_targets_per_im = reg_bin_targets_per_im[range(len(locations)),locations_to_gt_inds]

            reg_bin_targets_per_im[locations_to_min_area == INF,:] = 0 ##(20604,4)

            reg_bin_targets.append(reg_bin_targets_per_im)

            bin_range_batch.append(bin_range)

            # ####debug----------
            # l_bin_label,t_bin_label,b_bin_label,r_bin_label = reg_bin_targets_per_im[:,0], reg_bin_targets_per_im[:,1],reg_bin_targets_per_im[:,2],reg_bin_targets_per_im[:,3],
            # l_max,t_max,b_max,r_max = l_bin_label.max(), t_bin_label.max(), b_bin_label.max(),r_bin_label.max()
            # l_min, t_min, b_min, r_min = l_bin_label.min(), t_bin_label.min(), b_bin_label.min(), r_bin_label.min()
            # l_all_pixes_num_5 = sum(l_bin_label == 4)
            # t_all_pixes_num_5 = sum(t_bin_label == 4)
            # b_all_pixes_num_5 = sum(b_bin_label == 4)
            # r_all_pixes_num_5 = sum(r_bin_label == 4)

            # # #
            # # ###保存各bin分布结果
            # bin_num = num_bin
            # all_num_per_bin = np.zeros((4,bin_num))
            #
            # for i in range(4):
            #     # bin_dx = reg_bin_targets_per_im[:, i] - 1 ##由1开始转换为由0开始
            #     bin_dx = reg_bin_targets_per_im[:, i]
            #     for a_bin in range(bin_num):
            #         idx = bin_dx == (a_bin + 1)
            #         idx_sum = sum(idx)
            #         all_num_per_bin[i, a_bin] = idx_sum
            #
            #
            # # txt_path = '/data/piaozhengquan/projects/DAOD/MGADA-master/FCOS/bin_distri/bin_n10_city.txt'
            # # txt_path = '/data/piaozhengquan/projects/DAOD/MGADA-master/FCOS/bin_distri/bin_n10_kitti.txt'
            # txt_path = '/data/piaozhengquan/projects/DAOD/MGADA-master/FCOS/bin_distri/bin_n10_sim10k.txt'
            #
            # if not os.path.exists(txt_path):
            #     np.savetxt(txt_path, all_num_per_bin, fmt='%d')
            #
            # else:##读取
            #     last_all_num_per_bin = np.loadtxt(txt_path)
            #     all_num_per_bin += last_all_num_per_bin
            #     np.savetxt(txt_path, all_num_per_bin, fmt='%d')


        return labels, reg_targets, reg_bin_targets, bin_range_batch

    def compute_centerness_targets(self, reg_targets):
        # print('reg_targets: ', reg_targets.size())
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, box_bin, targets, box_regression_coarse):
        """
        Arguments:
            locations (list[BoxList]) ##list(5):(92x168,2)...特征图像各点投射到原输入图像上的坐标
            box_cls (list[Tensor])  ##list(5):(batch,1,92,168)...
            box_regression (list[Tensor])  ##list(5):(batch,4,92,168)(,,46,84)(,,23,42)(,,12,21)(,,6,11)
            centerness (list[Tensor])  ##list(5):(batch,1,92,168)...
            targets (list[BoxList])  ##list(batch_size)(实际图像坐标

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0) ##  batch_size
        num_classes = box_cls[0].size(1)
        num_bin = self.num_bin

        labels, reg_targets, reg_bin_targets, bin_range = self.prepare_targets(locations, targets, num_bin)

        ###labels(92376)(23184)(5796)(1512)(396)  92376 =  15456 * 6
        ###reg_targets(92376,4)(23184,4)(5796,4)(1512,4)(396,4)

        # print('labels[0].size(0): ', labels[0].size(0))
        # print('reg_targets[0].size(0): ', reg_targets[0].size(0))

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []

        box_bin_flatten = []

        labels_flatten = []
        reg_targets_flatten = []
        l_bin_flatten, t_bin_flatten, b_bin_flatten, r_bin_flatten = [],[],[],[]
        l_targets_bin_flatten, t_targets_bin_flatten, b_targets_bin_flatten, r_targets_bin_flatten = [], [], [], []

        bin_range_flatten = []

        box_regression_coarse_flatten = []

        for l in range(len(labels)):   ##5层 box_cls[l]:(bt,cls.eg.8,h,w);box_regression[l]:(bt,4,h,w);需要box_bin[l]:(bt,4*num_bin,h,w)
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))  ##centerness:(bt,1,h,w)

            labels_flatten.append(labels[l].reshape(-1))   ##labels[l]: size 1
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4)) ##reg_targets[l]: (xx,4)

            bin_range_flatten.append(bin_range[l].reshape(-1))

            box_regression_coarse_flatten.append(box_regression_coarse[l].permute(0, 2, 3, 1).reshape(-1, 4))

            ##----------l,t,b,r---------------
            l_targets_bin_flatten.append(reg_bin_targets[l][:, 0].reshape(-1))
            t_targets_bin_flatten.append(reg_bin_targets[l][:, 1].reshape(-1))
            b_targets_bin_flatten.append(reg_bin_targets[l][:, 2].reshape(-1))
            r_targets_bin_flatten.append(reg_bin_targets[l][:, 3].reshape(-1))

                ##预测输出格式：需要box_bin[l]: (bt, 4 * num_bin, h, w)
            l_bin_flatten.append(box_bin[l][:,0:num_bin,:,:].permute(0, 2, 3, 1).reshape(-1, num_bin))
            t_bin_flatten.append(box_bin[l][:, num_bin:2*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))
            b_bin_flatten.append(box_bin[l][:, 2*num_bin:3*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))
            r_bin_flatten.append(box_bin[l][:, 3*num_bin:4*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))


        box_cls_flatten = torch.cat(box_cls_flatten, dim=0) ##(123624,8=cls)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)  ##（123624，4）
        centerness_flatten = torch.cat(centerness_flatten, dim=0)  ##（123624）  size:1
        labels_flatten = torch.cat(labels_flatten, dim=0) ##（123624） size:1
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0) ##（123624，4）

        box_regression_coarse_flatten = torch.cat(box_regression_coarse_flatten, dim=0)

        boxes_iou = self.IoU(box_regression_coarse_flatten, reg_targets_flatten)
        label_mask = (boxes_iou > self.iou_th).long()
        labels_flatten = labels_flatten.mul(label_mask)


        bin_range_flatten = torch.cat(bin_range_flatten, dim=0)

        l_bin_flatten = torch.cat(l_bin_flatten, dim=0)
        t_bin_flatten = torch.cat(t_bin_flatten, dim=0)
        b_bin_flatten = torch.cat(b_bin_flatten, dim=0)
        r_bin_flatten = torch.cat(r_bin_flatten, dim=0)

        l_targets_bin_flatten = torch.cat(l_targets_bin_flatten, dim=0)
        t_targets_bin_flatten = torch.cat(t_targets_bin_flatten, dim=0)
        b_targets_bin_flatten = torch.cat(b_targets_bin_flatten, dim=0)
        r_targets_bin_flatten = torch.cat(r_targets_bin_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        ###-------------l,t,b,r bin分类损失-----------------
        # l_bin_flatten, l_targets_bin_flatten = l_bin_flatten[pos_inds], l_targets_bin_flatten[pos_inds]  ##(979) 变化
        # t_bin_flatten, t_targets_bin_flatten = t_bin_flatten[pos_inds], t_targets_bin_flatten[pos_inds] ##(979) 变化
        # b_bin_flatten, b_targets_bin_flatten = b_bin_flatten[pos_inds], b_targets_bin_flatten[pos_inds]  ##(979) 变化
        # r_bin_flatten, r_targets_bin_flatten = r_bin_flatten[pos_inds], r_targets_bin_flatten[pos_inds] ##(979) 变化

        cls_loss_l = self.cls_loss_func(
            l_bin_flatten,
            l_targets_bin_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        cls_loss_t = self.cls_loss_func(
            t_bin_flatten,
            t_targets_bin_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        cls_loss_b = self.cls_loss_func(
            b_bin_flatten,
            b_targets_bin_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        cls_loss_r = self.cls_loss_func(
            r_bin_flatten,
            r_targets_bin_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        bin_loss = (cls_loss_l + cls_loss_t + cls_loss_b + cls_loss_r) / 4
        # bin_loss = 0

        box_regression_flatten = box_regression_flatten[pos_inds]  ##(979) 变化
        reg_targets_flatten = reg_targets_flatten[pos_inds]  ##(979,4)
        centerness_flatten = centerness_flatten[pos_inds]  ##(979)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else: ##没有前景目标，输出损失取为预测输出的和
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        # ###-----------利用bin_target对box_reg进行一致性约束--------------
        # l_targets_bin_flatten =  l_targets_bin_flatten[pos_inds]  ##(979) 变化
        # t_targets_bin_flatten =  t_targets_bin_flatten[pos_inds] ##(979) 变化
        # b_targets_bin_flatten =  b_targets_bin_flatten[pos_inds]  ##(979) 变化
        # r_targets_bin_flatten =  r_targets_bin_flatten[pos_inds] ##(979) 变化
        #
        # delt_bin = bin_range_flatten[pos_inds] / num_bin  ##the width of each bin
        #
        # l_reg_bin = (box_regression_flatten[:,0] // delt_bin).int() + 1  ###the label of the first bin is 1
        # t_reg_bin = (box_regression_flatten[:, 1] // delt_bin).int() + 1  ###the label of the first bin is 1
        # b_reg_bin = (box_regression_flatten[:, 2] // delt_bin).int() + 1  ###the label of the first bin is 1
        # r_reg_bin = (box_regression_flatten[:, 3] // delt_bin).int() + 1  ###the label of the first bin is 1
        #
        # l_reg_bin[l_reg_bin < 0] = 0
        # l_reg_bin[l_reg_bin > num_bin] = num_bin
        # t_reg_bin[t_reg_bin < 0] = 0
        # t_reg_bin[t_reg_bin > num_bin] = num_bin
        # b_reg_bin[b_reg_bin < 0] = 0
        # b_reg_bin[b_reg_bin > num_bin] = num_bin
        # r_reg_bin[r_reg_bin < 0] = 0
        # r_reg_bin[r_reg_bin > num_bin] = num_bin
        #
        # ###曼哈顿距离
        # # reg_to_bin_loss_l = (l_reg_bin - l_targets_bin_flatten).abs().float().mean()
        # # reg_to_bin_loss_t = (t_reg_bin - t_targets_bin_flatten).abs().float().mean()
        # # reg_to_bin_loss_b = (b_reg_bin - b_targets_bin_flatten).abs().float().mean()
        # # reg_to_bin_loss_r = (r_reg_bin - r_targets_bin_flatten).abs().float().mean()
        #
        # ###汉明距离
        # reg_to_bin_loss_l = sum(l_reg_bin != l_targets_bin_flatten) / (pos_inds.numel() + N)
        # reg_to_bin_loss_t = sum(t_reg_bin != t_targets_bin_flatten) / (pos_inds.numel() + N)
        # reg_to_bin_loss_b = sum(b_reg_bin != b_targets_bin_flatten) / (pos_inds.numel() + N)
        # reg_to_bin_loss_r = sum(r_reg_bin != r_targets_bin_flatten) / (pos_inds.numel() + N)
        #
        # reg_to_bin_loss = (reg_to_bin_loss_l+reg_to_bin_loss_t+reg_to_bin_loss_b+reg_to_bin_loss_r) / 4
        #
        # # reg_loss += reg_to_bin_loss
        # reg_to_bin_loss_lam = 0.01
        # reg_loss = reg_loss + reg_to_bin_loss_lam * reg_to_bin_loss

        # bin_loss_lam = 0.1
        bin_loss_lam = 0.001
        bin_loss *= bin_loss_lam

        return cls_loss, reg_loss, centerness_loss, bin_loss

    def IoU(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        return gious


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

class FCOSLossComputation_adv(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_bin = cfg.BIN.NUM_REG_BIN
        self.center_aware_weight = cfg.MODEL.ADV.CENTER_AWARE_WEIGHT
        self.class_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.ca_dis_lambda = cfg.MODEL.ADV.CA_DIS_LAMBDA
        self.ga_dis_lambda = cfg.MODEL.ADV.GA_DIS_LAMBDA
        self.use_dis_global = cfg.MODEL.ADV.USE_DIS_GLOBAL
        self.use_dis_ca = cfg.MODEL.ADV.USE_DIS_CENTER_AWARE



    def prepare_targets(self, points, targets, num_bin):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)##(20604,2)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, reg_bin_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, num_bin
        )  ###所有层上点的坐标揉在一块  20604个（15456+3864+966+252+66）

        for i in range(len(labels)): ###per batch
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            reg_bin_targets[i] = torch.split(reg_bin_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        reg_bin_targets_level_first = []

        for level in range(len(points)):  ###5 feature maps num
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )  ###将各batch按feature map cat起来
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )
            reg_bin_targets_level_first.append(
                torch.cat([reg_bin_targets_per_im[level] for reg_bin_targets_per_im in reg_bin_targets], dim=0)
            )
        ###labels_level_first(92376)(23184)(5796)(1512)(396)  92376 =  15456 * 6
        ###reg_targets_level_first(92376,4)(23184,4)(5796,4)(1512,4)(396,4)
        # print('labels_level_first.size: ', labels_level_first[0].size())
        # print('reg_targets_level_first.size: ', reg_targets_level_first[0].size())
        return labels_level_first, reg_targets_level_first, reg_bin_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest,num_bin):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        reg_bin_targets = []  ##（20604,4)  not one-hot
        # num_bin = 10
        bin_range = object_sizes_of_interest[:, [1]]    ##(20604,1)
        # bin_range = object_sizes_of_interest[:, 0]  ##(20604)
        bin_range[bin_range == INF] = 1024

        for im_i in range(len(targets)):  ##per img in a batch
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox  ###(n,4)  n个bbox
            labels_per_im = targets_per_im.get_field("labels")  ##从1开始
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]  ##(20604,bbox_num_per_img)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)  ##(20604, bbox_num_per_img, 4)注意是stack不是cat

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF  ##(20604,bbox_num_per_img)

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds] ##(20604,4) #之前是stack
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)   ##list
            reg_targets.append(reg_targets_per_im)

            ###-------------对l,t,r,b的bin进行编码-------------
            delt_bin = bin_range / num_bin  ##the width of each bin
            l_bin_label = (l // delt_bin).int() + 1  ###the label of the first bin is 1
            t_bin_label = (t // delt_bin).int() + 1
            b_bin_label = (b // delt_bin).int() + 1
            r_bin_label = (r // delt_bin).int() + 1

            ###超过512+512的均放在最后一个bin
            l_bin_label[l_bin_label > num_bin] = num_bin
            t_bin_label[t_bin_label > num_bin] = num_bin
            b_bin_label[b_bin_label > num_bin] = num_bin
            r_bin_label[r_bin_label > num_bin] = num_bin

            reg_bin_targets_per_im = torch.stack([l_bin_label, t_bin_label, b_bin_label, r_bin_label], dim=2)

            reg_bin_targets_per_im = reg_bin_targets_per_im[range(len(locations)),locations_to_gt_inds]

            reg_bin_targets_per_im[locations_to_min_area == INF,:] = 0

            reg_bin_targets.append(reg_bin_targets_per_im)

            # ####debug----------
            # l_bin_label,t_bin_label,b_bin_label,r_bin_label = reg_bin_targets_per_im[:,0], reg_bin_targets_per_im[:,1],reg_bin_targets_per_im[:,2],reg_bin_targets_per_im[:,3],
            # l_max,t_max,b_max,r_max = l_bin_label.max(), t_bin_label.max(), b_bin_label.max(),r_bin_label.max()
            # l_min, t_min, b_min, r_min = l_bin_label.min(), t_bin_label.min(), b_bin_label.min(), r_bin_label.min()
            # l_all_pixes_num_5 = sum(l_bin_label == 4)
            # t_all_pixes_num_5 = sum(t_bin_label == 4)
            # b_all_pixes_num_5 = sum(b_bin_label == 4)
            # r_all_pixes_num_5 = sum(r_bin_label == 4)


        return labels, reg_targets, reg_bin_targets

    def compute_centerness_targets(self, reg_targets):
        # print('reg_targets: ', reg_targets.size())
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, box_cls_source,  centerness_source, box_bin_source, adv_logits_source, adv_bbox_bin_source,
            box_cls_target, centerness_target, box_bin_target, adv_logits_target, adv_bbox_bin_target,
            targets):

        """
        Arguments:
            locations (list[BoxList]) ##list(5):(92x168,2)...特征图像各点投射到原输入图像上的坐标
            box_cls (list[Tensor])  ##list(5):(batch,1,92,168)...
            box_regression (list[Tensor])  ##list(5):(batch,4,92,168)(,,46,84)(,,23,42)(,,12,21)(,,6,11)
            centerness (list[Tensor])  ##list(5):(batch,1,92,168)...
            targets (list[BoxList])  ##list(batch_size)(实际图像坐标

        Returns:
            cls_loss_adv (Tensor)
            loss_centerness_adv (Tensor)
            loss_bin_adv (Tensor)
        """
        # N = box_cls[0].size(0) ##  batch_size
        # num_classes = box_cls[0].size(1)
        # num_bin = self.num_bin

        # labels, reg_targets, reg_bin_targets = self.prepare_targets(locations, targets, num_bin)
        #
        # ###labels(92376)(23184)(5796)(1512)(396)  92376 =  15456 * 6
        # ###reg_targets(92376,4)(23184,4)(5796,4)(1512,4)(396,4)
        #
        # # print('labels[0].size(0): ', labels[0].size(0))
        # # print('reg_targets[0].size(0): ', reg_targets[0].size(0))
        #
        # box_cls_flatten = []
        # box_regression_flatten = []
        # centerness_flatten = []
        #
        # box_bin_flatten = []
        #
        # labels_flatten = []
        # reg_targets_flatten = []
        # l_bin_flatten, t_bin_flatten, b_bin_flatten, r_bin_flatten = [],[],[],[]
        # l_targets_bin_flatten, t_targets_bin_flatten, b_targets_bin_flatten, r_targets_bin_flatten = [], [], [], []
        #
        #
        # for l in range(len(labels)):   ##5层 box_cls[l]:(bt,cls.eg.8,h,w);box_regression[l]:(bt,4,h,w);需要box_bin[l]:(bt,4*num_bin,h,w)
        #     box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
        #     box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
        #     centerness_flatten.append(centerness[l].reshape(-1))  ##centerness:(bt,1,h,w)
        #
        #     labels_flatten.append(labels[l].reshape(-1))   ##labels[l]: size 1
        #     reg_targets_flatten.append(reg_targets[l].reshape(-1, 4)) ##reg_targets[l]: (xx,4)
        #
        #     ##----------l,t,b,r---------------
        #     l_targets_bin_flatten.append(reg_bin_targets[l][:, 0].reshape(-1))
        #     t_targets_bin_flatten.append(reg_bin_targets[l][:, 1].reshape(-1))
        #     b_targets_bin_flatten.append(reg_bin_targets[l][:, 2].reshape(-1))
        #     r_targets_bin_flatten.append(reg_bin_targets[l][:, 3].reshape(-1))
        #
        #         ##预测输出格式：需要box_bin[l]: (bt, 4 * num_bin, h, w)
        #     l_bin_flatten.append(box_bin[l][:,0:num_bin,:,:].permute(0, 2, 3, 1).reshape(-1, num_bin))
        #     t_bin_flatten.append(box_bin[l][:, num_bin:2*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))
        #     b_bin_flatten.append(box_bin[l][:, 2*num_bin:3*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))
        #     r_bin_flatten.append(box_bin[l][:, 3*num_bin:4*num_bin, :, :].permute(0, 2, 3, 1).reshape(-1, num_bin))
        #
        #
        # box_cls_flatten = torch.cat(box_cls_flatten, dim=0) ##(123624,8=cls)
        # box_regression_flatten = torch.cat(box_regression_flatten, dim=0)  ##（123624，4）
        # centerness_flatten = torch.cat(centerness_flatten, dim=0)  ##（123624）  size:1
        # labels_flatten = torch.cat(labels_flatten, dim=0) ##（123624） size:1
        # reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0) ##（123624，4）
        #
        # l_bin_flatten = torch.cat(l_bin_flatten, dim=0)
        # t_bin_flatten = torch.cat(t_bin_flatten, dim=0)
        # b_bin_flatten = torch.cat(b_bin_flatten, dim=0)
        # r_bin_flatten = torch.cat(r_bin_flatten, dim=0)
        #
        # l_targets_bin_flatten = torch.cat(l_targets_bin_flatten, dim=0)
        # t_targets_bin_flatten = torch.cat(t_targets_bin_flatten, dim=0)
        # b_targets_bin_flatten = torch.cat(b_targets_bin_flatten, dim=0)
        # r_targets_bin_flatten = torch.cat(r_targets_bin_flatten, dim=0)
        #
        # pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten,
        #     labels_flatten.int()
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        #
        # ###-------------l,t,b,r bin分类损失-----------------
        # l_bin_flatten, l_targets_bin_flatten = l_bin_flatten[pos_inds], l_targets_bin_flatten[pos_inds]  ##(979) 变化
        # t_bin_flatten, t_targets_bin_flatten = t_bin_flatten[pos_inds], t_targets_bin_flatten[pos_inds] ##(979) 变化
        # b_bin_flatten, b_targets_bin_flatten = b_bin_flatten[pos_inds], b_targets_bin_flatten[pos_inds]  ##(979) 变化
        # r_bin_flatten, r_targets_bin_flatten = r_bin_flatten[pos_inds], r_targets_bin_flatten[pos_inds] ##(979) 变化
        #
        # cls_loss_l = self.cls_loss_func(
        #     l_bin_flatten,
        #     l_targets_bin_flatten.int()
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        # cls_loss_t = self.cls_loss_func(
        #     t_bin_flatten,
        #     t_targets_bin_flatten.int()
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        # cls_loss_b = self.cls_loss_func(
        #     b_bin_flatten,
        #     b_targets_bin_flatten.int()
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        # cls_loss_r = self.cls_loss_func(
        #     r_bin_flatten,
        #     r_targets_bin_flatten.int()
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        #
        # bin_loss = (cls_loss_l + cls_loss_t + cls_loss_b + cls_loss_r) / 4
        # # bin_loss = 0
        #
        # box_regression_flatten = box_regression_flatten[pos_inds]  ##(979) 变化
        # reg_targets_flatten = reg_targets_flatten[pos_inds]  ##(979,4)
        # centerness_flatten = centerness_flatten[pos_inds]  ##(979)
        #
        # if pos_inds.numel() > 0:
        #     centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
        #     reg_loss = self.box_reg_loss_func(
        #         box_regression_flatten,
        #         reg_targets_flatten,
        #         centerness_targets
        #     )
        #     centerness_loss = self.centerness_loss_func(
        #         centerness_flatten,
        #         centerness_targets
        #     )
        # else: ##没有前景目标，输出损失取为预测输出的和
        #     reg_loss = box_regression_flatten.sum()
        #     centerness_loss = centerness_flatten.sum()

        ##------------------对抗判别损失--------------------
        loss_adv_box_cls, loss_adv_centerness, loss_adv_box_bin = 0,0,0
            ####box_cls:  box_cls, adv_box_cls
        for l in range(len(box_cls_source)):

            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ####box_bin:      ++++++++++++++++++++++++++++++++
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ###----------------attention Map---------------------
            # Generate cneter-aware map
            ###源域
            # loss_adv_box_bin_l, loss_adv_box_bin_t, loss_adv_box_bin_b, loss_adv_box_bin_r = 0,0,0,0
            ###预测器中l,t,b,r分别处理
            box_bin_source_level = box_bin_source[l]  ##batch, 4*bin,h,w
            num_reg_bin = int(box_bin_source_level.size(1) / 4)
            box_bin_source_level_l, box_bin_source_level_t,box_bin_source_level_b,box_bin_source_level_r\
                = box_bin_source_level[:,:num_reg_bin,:,:], box_bin_source_level[:,num_reg_bin:2*num_reg_bin,:,:],box_bin_source_level[:,2*num_reg_bin:3*num_reg_bin,:,:]\
                ,box_bin_source_level[:,3*num_reg_bin:4*num_reg_bin,:,:]

            # box_bin_source_level_l, box_bin_source_level_t, box_bin_source_level_b, box_bin_source_level_r =\
            #     F.softmax(box_bin_source_level_l,dim=1), F.softmax(box_bin_source_level_t,dim=1), \
            #     F.softmax(box_bin_source_level_b,dim=1), F.softmax(box_bin_source_level_r,dim=1)

            ###对抗预测器adv中l,t,b,r分别处理
            adv_box_bin_source_level_l, adv_box_bin_source_level_t, adv_box_bin_source_level_b, adv_box_bin_source_level_r \
                = adv_bbox_bin_source[l][:, :num_reg_bin, :, :], adv_bbox_bin_source[l][:, num_reg_bin:2 * num_reg_bin, :,:],\
                  adv_bbox_bin_source[l][:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
                , adv_bbox_bin_source[l][:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

            box_bin_source_level_l_argmax, box_bin_source_level_t_argmax, box_bin_source_level_b_argmax, box_bin_source_level_r_argmax\
                = box_bin_source_level_l.max(1)[1],box_bin_source_level_t.max(1)[1],box_bin_source_level_b.max(1)[1],box_bin_source_level_r.max(1)[1]

            ###目标域
            ###预测器中l,t,b,r分别处理
            box_bin_target_level = box_bin_target[l]  ##batch, 4*bin,h,w
            box_bin_target_level_l, box_bin_target_level_t, box_bin_target_level_b, box_bin_target_level_r \
                = box_bin_target_level[:, :num_reg_bin, :, :], box_bin_target_level[:, num_reg_bin:2 * num_reg_bin, :,:], \
                  box_bin_target_level[:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
                , box_bin_target_level[:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

            # box_bin_target_level_l, box_bin_target_level_t, box_bin_target_level_b, box_bin_target_level_r = \
            #     F.softmax(box_bin_target_level_l, dim=1), F.softmax(box_bin_target_level_t, dim=1), \
            #     F.softmax(box_bin_target_level_b, dim=1), F.softmax(box_bin_target_level_r, dim=1)

            ###对抗预测器adv中l,t,b,r分别处理
            adv_box_bin_target_level_l, adv_box_bin_target_level_t, adv_box_bin_target_level_b, adv_box_bin_target_level_r \
                = adv_bbox_bin_target[l][:, :num_reg_bin, :, :], adv_bbox_bin_target[l][:, num_reg_bin:2 * num_reg_bin,:, :], \
                  adv_bbox_bin_target[l][:, 2 * num_reg_bin:3 * num_reg_bin, :, :] \
                , adv_bbox_bin_target[l][:, 3 * num_reg_bin:4 * num_reg_bin, :, :]

            box_bin_target_level_l_argmax, box_bin_target_level_t_argmax, box_bin_target_level_b_argmax, box_bin_target_level_r_argmax \
                = box_bin_target_level_l.max(1)[1], box_bin_target_level_t.max(1)[1], box_bin_target_level_b.max(1)[1], \
                  box_bin_target_level_r.max(1)[1]

            # adv_box_bin_l_log_source = torch.log(_sigmoid(adv_bbox_bin_source[l]))
            # adv_box_bin_l_log_source = adv_box_bin_l_log_source * atten_map_source
            # loss_adv_box_bin_source = F.nll_loss(adv_box_bin_l_log_source, box_bin_source_argmax)

            # adv_box_bin_t_log_source = torch.log(F.softmax(adv_box_bin_source_level_t, dim=1))
            # adv_box_bin_t_log_source = adv_box_bin_t_log_source * atten_map_source
            # loss_adv_box_bin_source_t = F.nll_loss(adv_box_bin_t_log_source, box_bin_source_level_t_argmax)

            adv_box_bin_l_log_source = torch.log(adv_box_bin_source_level_l.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_l_log_source = adv_box_bin_l_log_source * atten_map_source
            loss_adv_box_bin_source_l = F.nll_loss(adv_box_bin_l_log_source, box_bin_source_level_l_argmax)

            adv_box_bin_l_log_target = torch.log(1-adv_box_bin_target_level_l.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_l_log_target = adv_box_bin_l_log_target * atten_map_target
            loss_adv_box_bin_target_l = F.nll_loss(adv_box_bin_l_log_target, box_bin_target_level_l_argmax)

            loss_adv_box_bin_l = loss_adv_box_bin_source_l + loss_adv_box_bin_target_l
            ##--------------------t
            adv_box_bin_t_log_source = torch.log(adv_box_bin_source_level_t.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_t_log_source = adv_box_bin_t_log_source * atten_map_source
            loss_adv_box_bin_source_t = F.nll_loss(adv_box_bin_t_log_source, box_bin_source_level_t_argmax)

            adv_box_bin_t_log_target = torch.log(1 - adv_box_bin_target_level_t.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_t_log_target = adv_box_bin_t_log_target * atten_map_target
            loss_adv_box_bin_target_t = F.nll_loss(adv_box_bin_t_log_target, box_bin_target_level_t_argmax)

            loss_adv_box_bin_t = loss_adv_box_bin_source_t + loss_adv_box_bin_target_t
            ##--------------------b
            adv_box_bin_b_log_source = torch.log(adv_box_bin_source_level_b.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_b_log_source = adv_box_bin_b_log_source * atten_map_source
            loss_adv_box_bin_source_b = F.nll_loss(adv_box_bin_b_log_source, box_bin_source_level_b_argmax)

            adv_box_bin_b_log_target = torch.log(1 - adv_box_bin_target_level_b.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_b_log_target = adv_box_bin_b_log_target * atten_map_target
            loss_adv_box_bin_target_b = F.nll_loss(adv_box_bin_b_log_target, box_bin_target_level_b_argmax)

            loss_adv_box_bin_b = loss_adv_box_bin_source_b + loss_adv_box_bin_target_b
            ##--------------------r
            adv_box_bin_r_log_source = torch.log(adv_box_bin_source_level_r.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_r_log_source = adv_box_bin_r_log_source * atten_map_source
            loss_adv_box_bin_source_r = F.nll_loss(adv_box_bin_r_log_source, box_bin_source_level_r_argmax)

            adv_box_bin_r_log_target = torch.log(1 - adv_box_bin_target_level_r.sigmoid())
            # if self.use_dis_ca:
            #     adv_box_bin_r_log_target = adv_box_bin_r_log_target * atten_map_target
            loss_adv_box_bin_target_r = F.nll_loss(adv_box_bin_r_log_target, box_bin_target_level_r_argmax)

            loss_adv_box_bin_r = loss_adv_box_bin_source_r + loss_adv_box_bin_target_r
            ##-----------------------------------------------------------------------------------
            loss_adv_box_bin_ltbr = (loss_adv_box_bin_l + loss_adv_box_bin_t + loss_adv_box_bin_b + loss_adv_box_bin_r)/4


            loss_adv_box_bin +=  loss_adv_box_bin_ltbr

        num_layers = len(box_cls_source)
        # loss_adv_box_cls, loss_adv_centerness, loss_adv_box_bin = loss_adv_box_cls / num_layers, loss_adv_centerness  / num_layers, loss_adv_box_bin  / num_layers
        # return loss_adv_box_cls, loss_adv_centerness, loss_adv_box_bin

        # loss_adv_box_cls, loss_adv_box_bin = loss_adv_box_cls / num_layers, loss_adv_box_bin / num_layers

        if self.use_dis_ca: ##with CA
            loss_adv_box_cls, loss_adv_box_bin = self.ca_dis_lambda * loss_adv_box_cls, self.ca_dis_lambda * loss_adv_box_bin
        else:  ##GA only
            loss_adv_box_cls, loss_adv_box_bin = self.ga_dis_lambda * loss_adv_box_cls, self.ga_dis_lambda * loss_adv_box_bin

        return loss_adv_box_cls, loss_adv_box_bin

def make_fcos_loss_evaluator_adv(cfg):
    loss_evaluator = FCOSLossComputation_adv(cfg)
    return loss_evaluator