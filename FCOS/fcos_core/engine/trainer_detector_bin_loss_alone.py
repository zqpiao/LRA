# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import cv2

import torch
import torch.nn as nn
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list
im_index = 0

import os
from fcos_core.utils.comm import synchronize
from fcos_core.data import make_data_loader
# from fcos_core.engine.inference import inference
# from fcos_core.engine.inference_detector import inference

from fcos_core.utils.miscellaneous import mkdir
import copy
from fcos_core.utils.imports import import_file


from tqdm import tqdm

from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


# def foward_detector(model, images, targets=None, return_maps=False):
#     map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
#     feature_layers = map_layer_to_index.keys()
#     if "genbox" in model.keys() and "genfeature" in model.keys():
#        use_wlm = True
#     else:
#        use_wlm = False
#
#     model_backbone = model["backbone"]
#     if "genbox" in model.keys():
#         model_genbox = model["genbox"]
#     if "genfeature" in model.keys():
#         model_genfeature = model["genfeature"]
#     model_fcos = model["fcos"]
#
#     images = to_image_list(images)
#     dict_features = model_backbone(images.tensors)
#     pre_features, features = dict_features['pre_features'], dict_features['features']
#
#
#     #local feature
#     f_dt = {
#         layer: features[map_layer_to_index[layer]]
#         for layer in feature_layers
#     }
#     losses = {}
#
#
#     if model_fcos.training and targets is None:
#         # train G on target domain
#         if use_wlm:
#             _, detector_loss, detector_maps = model_genbox(images, features, targets=None, return_maps=return_maps)
#             features_gl = model_genfeature(features, detector_maps['box_regression'], images.tensors.size(), targets=None, return_maps=return_maps)
#             proposals, proposal_losses, score_maps = model_fcos(
#                 images, features_gl, targets=None, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'])
#         else:
#             proposals, proposal_losses, score_maps = model_fcos(
#                 images, features, targets=None, return_maps=return_maps)
#         assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0  # loss_dict should be empty dict
#     else:
#         # train G on source domain / inference
#         if use_wlm:
#             _, detector_loss, detector_maps = model_genbox(images, features, targets=targets, return_maps=return_maps)
#             features_gl = model_genfeature(features, detector_maps['box_regression'], images.tensors.size(), targets=targets, return_maps=return_maps)
#
#             proposals, proposal_losses, score_maps = model_fcos(
#                 images, features_gl, targets=targets, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'])
#
#
#         else:
#             proposals, proposal_losses, score_maps = model_fcos(
#                 images, features, targets=targets, return_maps=return_maps)
#
#     #global feature
#     if use_wlm:
#         f_gl = {
#             layer: features_gl[map_layer_to_index[layer]]
#             for layer in feature_layers
#         }
#     losses = {}
#
#     if model_fcos.training:
#         # training
#         m = {
#             layer: {
#                 map_type:
#                 score_maps[map_type][map_layer_to_index[layer]]
#                 for map_type in score_maps
#             }
#             for layer in feature_layers
#         }
#         losses.update(proposal_losses)
#         if use_wlm:
#             losses.update(detector_loss)
#             return losses, f_dt, f_gl, m
#         else:
#             return losses, f_dt, f_dt, m
#     else:
#         # inference
#         result = proposals
#         return result

def compute_on_dataset(model, data_loader, device, timer=None):
    # model.eval
    for k in model:
        model[k].eval()

    results_dict = {}
    cpu_device = torch.device("cpu")
    img_index = 0
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = foward_detector(model, images, None, targets=None)

            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions
def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

def foward_detector(model, images_source, images_target, targets=None, return_maps=False):
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()
    if "genbox" in model.keys() and "genfeature" in model.keys():
        use_wlm = True
    else:
        use_wlm = False

    model_backbone = model["backbone"]
    if "genbox" in model.keys():
        model_genbox = model["genbox"]
    if "genfeature" in model.keys():
        model_genfeature = model["genfeature"]
    model_fcos = model["fcos"]

    if (not model_fcos.training) or (model_fcos.training and targets is not None):  ###仅在source上进行有监督训练 或者测试阶段
        images = to_image_list(images_source)
        dict_features = model_backbone(images.tensors)

        pre_features, features = dict_features['pre_features'], dict_features['features']

        # local feature
        f_dt = {
            layer: features[map_layer_to_index[layer]]
            for layer in feature_layers
        }

        # f = {
        #     layer: features[map_layer_to_index[layer]]
        #     for layer in feature_layers
        # }
        losses = {}
        # print("hehe")
        if (not model_fcos.training) and targets is None:  ##测试阶段

            if use_wlm:
                _, detector_loss, detector_maps = model_genbox(images, features, targets=targets,
                                                               return_maps=return_maps)
                features_gl = model_genfeature(features, detector_maps['box_regression'], images.tensors.size(),
                                               targets=targets, return_maps=return_maps)

                proposals, proposal_losses, _ = model_fcos(
                    images, features_gl, None, None, targets=targets, return_maps=return_maps,
                    box_regression_coarse=detector_maps['box_regression'])


            else:
                proposals, proposal_losses,_, score_maps = model_fcos(
                    images, features, None, None, targets=targets, return_maps=return_maps)

        else: ###仅在source上进行有监督训练

            if use_wlm:
                _, detector_loss, detector_maps = model_genbox(images, features, targets=targets,
                                                               return_maps=return_maps)
                features_gl = model_genfeature(features, detector_maps['box_regression'], images.tensors.size(),
                                               targets=targets, return_maps=return_maps)

                proposals, proposal_losses, _, score_maps = model_fcos(
                    images, features_gl, None, None, targets=targets, return_maps=return_maps,
                    box_regression_coarse=detector_maps['box_regression'])


            else:
                proposals, proposal_losses, _, score_maps = model_fcos(
                    images, features, None, None, targets=targets, return_maps=return_maps)



        if model_fcos.training:
            # training
            # m = {
            #     layer: {
            #         map_type:
            #             score_maps[map_type][map_layer_to_index[layer]]
            #         for map_type in score_maps
            #     }
            #     for layer in feature_layers
            # }
            losses.update(proposal_losses)
            if use_wlm:
                losses.update(detector_loss)
            #     return losses, f_dt, f_gl, m
            # else:
            #     return losses, f_dt, f_dt, m
            # return losses, f, m
            return losses, None, None, None
        else:
            # inference
            result = proposals
            return result

    elif model_fcos.training and targets is None:  ###仅在source和target上进行对抗训练
        images_source = to_image_list(images_source) ##batch, 3, 6xx,13xx
        images_target = to_image_list(images_target) ##batch, 3, 6xx,13xx

        # features_source = model_backbone(images_source.tensors)
        # features_target = model_backbone(images_target.tensors)

        dict_features_source = model_backbone(images_source.tensors)
        pre_features_source, features_source = dict_features_source['pre_features'], dict_features_source['features']

        dict_features_target = model_backbone(images_target.tensors)
        pre_features_target, features_target = dict_features_target['pre_features'], dict_features_target['features']

        f_dt_s = {
            layer: features_source[map_layer_to_index[layer]]
            for layer in feature_layers
        }
        f_dt_t = {
            layer: features_target[map_layer_to_index[layer]]
            for layer in feature_layers
        }

        if use_wlm:
            _, detector_loss_source, detector_maps_source = model_genbox(images_source, features_source, targets=targets,
                                                           return_maps=return_maps)
            features_gl_source = model_genfeature(features_source, detector_maps_source['box_regression'], images_source.tensors.size(),
                                           targets=targets, return_maps=return_maps)

            _, detector_loss_target, detector_maps_target = model_genbox(images_target, features_target,
                                                                         targets=targets,
                                                                         return_maps=return_maps)
            features_gl_target = model_genfeature(features_target, detector_maps_target['box_regression'],
                                                  images_target.tensors.size(),
                                                  targets=targets, return_maps=return_maps)

            f_gl_s = {
                layer: features_gl_source[map_layer_to_index[layer]]
                for layer in feature_layers
            }
            f_gl_t = {
                layer: features_gl_target[map_layer_to_index[layer]]
                for layer in feature_layers
            }

            proposals, adv_proposal_losses, forward_results, score_maps = model_fcos(
                images_source, features_gl_source, images_target, features_gl_target, targets=None, return_maps=return_maps)

        else:
            f_gl_s = {
                layer: features_source[map_layer_to_index[layer]]
                for layer in feature_layers
            }
            f_gl_t = {
                layer: features_target[map_layer_to_index[layer]]
                for layer in feature_layers
            }

            proposals, adv_proposal_losses, forward_results, score_maps = model_fcos(
                images_source, features_source, images_target, features_target, targets=None, return_maps=return_maps)

        losses = {}
        # print("hehe")
        # if model_fcos.training and targets is None:
            # train G on source domain
        # proposals, adv_proposal_losses, forward_results, score_maps = model_fcos(
        #     images_source, features_source, images_target, features_target, targets=None, return_maps=return_maps)
        # assert len(adv_proposal_losses) == 1 and adv_proposal_losses["zero"] == 0  # loss_dict should be empty dict
        # else:
        #     # train G on source domain / inference
        #     proposals, proposal_losses, score_maps = model_fcos(
        #         images, features, targets=targets, return_maps=return_maps)

        # if model_fcos.training:
        # training
        score_maps_source, score_maps_target = score_maps[0], score_maps[1]
        m_s = {
            layer: {
                map_type:
                score_maps_source[map_type][map_layer_to_index[layer]]
                for map_type in score_maps_source
            }
            for layer in feature_layers
        }

        m_t = {
            layer: {
                map_type:
                    score_maps_target[map_type][map_layer_to_index[layer]]
                for map_type in score_maps_target
            }
            for layer in feature_layers
        }

        forward_results_0 = {
            layer: forward_results[0][map_layer_to_index[layer]]
            for layer in feature_layers
        }
        forward_results_1 = {
            layer: forward_results[1][map_layer_to_index[layer]]
            for layer in feature_layers
        }
        forward_results_2 = {
            layer: forward_results[2][map_layer_to_index[layer]]
            for layer in feature_layers
        }
        forward_results_3 = {
            layer: forward_results[3][map_layer_to_index[layer]]
            for layer in feature_layers
        }

        # losses.update(adv_proposal_losses)
        # return losses, f, m


        return losses, (forward_results_0,forward_results_1,forward_results_2,forward_results_3), (f_dt_s,f_dt_t, f_gl_s,f_gl_t), (m_s, m_t)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def run_test(cfg, model, distributed):
    if distributed:
        model["backbone"] = model["backbone"].module
        model["fcos"] = model["fcos"].module
        #if cfg.MODEL.ADV.USE_DIS_P7:
        #    model["dis_P7"] = model["dis_P7"].module
        #if cfg.MODEL.ADV.USE_DIS_P6:
        #    model["dis_P6"] = model["dis_P6"].module
        #if cfg.MODEL.ADV.USE_DIS_P5:
        #    model["dis_P5"] = model["dis_P5"].module
        #if cfg.MODEL.ADV.USE_DIS_P4:
        #    model["dis_P4"] = model["dis_P4"].module
        #if cfg.MODEL.ADV.USE_DIS_P3:
        #    model["dis_P3"] = model["dis_P3"].module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    map_0_5 = 0
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        val_pp = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

        paths_catalog = import_file(
            "fcos_core.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        DatasetCatalog = paths_catalog.DatasetCatalog
        data = DatasetCatalog.get(dataset_name)

        if data["factory"] == "COCODataset":
            if val_pp:
            # print('val_pp: ', val_pp)
                eval_str = str(val_pp[0])
                map_0_5 = float(eval_str.split(' ')[4][:-2])
            else:
                map_0_5 = 0.0
        else:
            # print('val_pp: ', val_pp['map'])
            map_0_5 = val_pp['map']
    return map_0_5


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        cfg,
        distributed,
        arguments,
):
    if "use_dis_global" in arguments.keys():
        USE_DIS_GLOBAL = arguments["use_dis_global"]
    
    if "use_dis_ca" in arguments.keys():
        USE_DIS_CENTER_AWARE = arguments["use_dis_ca"]
    
    if "use_feature_layers" in arguments.keys():
         used_feature_layers = arguments["use_feature_layers"]
    
    if "use_dis_detect_gl" in arguments.keys():
         USE_DIS_DETECT_GL = arguments["use_dis_detect_gl"]

    if "use_cm_global" in arguments.keys():
         USE_CM_GLOBAL = arguments["use_cm_global"]


    if "use_dis_bin" in arguments.keys():
        USE_DIS_BIN = arguments["use_dis_bin"]

    # dataloader
    data_loader_source = data_loader["source"]
    data_loader_target = data_loader["target"]

    # classified label of source domain and target domain
    source_label = 0.0
    target_label = 1.0

    # dis_lambda
    if USE_DIS_DETECT_GL:
        dt_dis_lambda = arguments["dt_dis_lambda"]
    if USE_DIS_GLOBAL:
        ga_dis_lambda = arguments["ga_dis_lambda"]
    if USE_DIS_CENTER_AWARE:
        ca_dis_lambda = arguments["ca_dis_lambda"]
    if USE_CM_GLOBAL:
        cm_dis_lambda = arguments["ga_cm_lambda"]

    if USE_DIS_BIN:
        bin_dis_lambda = arguments["bin_dis_lambda"]

    # Start training
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")

    # model.train()
    for k in model:
        model[k].train()

    meters = MetricLogger(delimiter="  ")
    assert len(data_loader_source) == len(data_loader_target)
    max_iter = max(len(data_loader_source), len(data_loader_target))
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    map_0_5_max = arguments["map_best"]

    for iteration, ((images_s, targets_s, _), (images_t, _, _)) \
        in enumerate(zip(data_loader_source, data_loader_target), start_iter):

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            for k in scheduler:
                scheduler[k].step()

        images_s = images_s.to(device)
        targets_s = [target_s.to(device) for target_s in targets_s]
        images_t = images_t.to(device)

        # optimizer.zero_grad()
        for k in optimizer:
            optimizer[k].zero_grad()

        ##########################################################################
        #################### (1): train G with source domain #####################
        ##########################################################################

        loss_dict, _, _, _ = foward_detector(
            model, images_s, None, targets=targets_s, return_maps=True)

        # loss_dict, features_lc_s, features_gl_s, score_maps_s = foward_detector(
        #     model, images_s, targets=targets_s, return_maps=True)

        # rename loss to indicate domain
        loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

        losses.backward(retain_graph=True)
        del loss_dict, losses

        ##########################################################################
        #################### (2): train D with source domain #####################
        ##########################################################################

        loss_dict, forward_results, features, score_maps = foward_detector(
            model, images_s, images_t, targets=None, return_maps=True)

        box_cls_source, box_cls_target, box_bin_source, box_bin_target = \
            forward_results[0], forward_results[1], forward_results[2], forward_results[3]

        features_lc_s, features_lc_t, features_gl_s, features_gl_t = features[0], features[1], features[2], features[3],

        score_maps_s, score_maps_t = score_maps[0], score_maps[1]


        loss_dict = {}
        for layer in used_feature_layers:
            # detatch score_map
            for map_type in score_maps_s[layer]:
                score_maps_s[layer][map_type] = score_maps_s[layer][map_type].detach()
            if USE_DIS_DETECT_GL:
                loss_dict["loss_detect_%s_ds" % layer] = \
                    dt_dis_lambda * model["d_dis_%s" % layer](features_lc_s[layer], source_label, domain='source')
            if USE_DIS_GLOBAL:
                loss_dict["loss_adv_%s_ds" % layer] = \
                    ga_dis_lambda * model["dis_%s" % layer](features_gl_s[layer], source_label, domain='source')
            if USE_DIS_CENTER_AWARE:
                loss_dict["loss_adv_%s_CA_ds" % layer] = \
                    ca_dis_lambda * model["dis_%s_CA" % layer](features_gl_s[layer], source_label, score_maps_s[layer], domain='source')
            if USE_CM_GLOBAL:
                loss_dict["loss_cm_%s_ds" % layer] = \
                    cm_dis_lambda * model["cm_%s" % layer](features_gl_s[layer], source_label, score_maps_s, targets_s, layer, domain='source')



            ###target
            for map_type in score_maps_t[layer]:
                score_maps_t[layer][map_type] = score_maps_t[layer][map_type].detach()

            if USE_DIS_DETECT_GL:
                loss_dict["loss_detect_%s_dt" % layer] = \
                    dt_dis_lambda * model["d_dis_%s" % layer](features_lc_t[layer], target_label, domain='target')

            if USE_DIS_GLOBAL:
                loss_dict["loss_adv_%s_dt" % layer] = \
                    ga_dis_lambda * model["dis_%s" % layer](features_gl_t[layer], target_label, domain='target')
            if USE_DIS_CENTER_AWARE:
                loss_dict["loss_adv_%s_CA_dt" % layer] = \
                    ca_dis_lambda * model["dis_%s_CA" % layer](features_gl_t[layer], target_label, score_maps_t[layer],
                                                               domain='target')
            if USE_CM_GLOBAL:
                loss_dict["loss_cm_%s_dt" % layer] = \
                    cm_dis_lambda * model["cm_%s" % layer](features_gl_t[layer], target_label, score_maps_t, None,
                                                           layer, domain='target')
                # print("t_loss_cm_%s_ds" % layer, ':', loss_dict["loss_cm_%s_ds" % layer])

            # if USE_DIS_BIN:
            #     ##gl feature
            #     loss_dict["loss_adv_%s_bin" % layer] = \
            #         bin_dis_lambda * model["dis_%s_BIN" % layer](box_bin_source[layer], box_bin_target[layer],
            #     features_gl_s[layer], features_gl_t[layer], score_maps_s[layer], score_maps_t[layer])

                # ##lc feature
                # loss_dict["loss_adv_%s_bin" % layer] = \
                #     bin_dis_lambda * model["dis_%s_BIN" % layer](box_bin_source[layer], box_bin_target[layer],
                #                                                  features_lc_s[layer], features_lc_t[layer],
                #                                                  score_maps_s[layer], score_maps_t[layer]
                #                                                 )



        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

        losses.backward(retain_graph=True)
        del loss_dict, losses


        ####bin-loss--------------------------------------------
        loss_dict = {}
        for layer in used_feature_layers:
            if USE_DIS_BIN:
                ##gl feature
                loss_dict["loss_adv_%s_bin" % layer] = \
                    bin_dis_lambda * model["dis_%s_BIN" % layer](box_bin_source[layer], box_bin_target[layer],
                                                             features_gl_s[layer], features_gl_t[layer],
                                                             score_maps_s[layer], score_maps_t[layer])

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

        losses.backward(retain_graph=True)
        del loss_dict, losses



            # # reduce losses over all GPUs for logging purposes
        # if loss_dict:
        #     #print(loss_dict.keys())
        #     loss_dict_reduced = reduce_loss_dict(loss_dict)
        #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #     meters.update(loss_ds=losses_reduced, **loss_dict_reduced)
        #
        #     losses.backward()
        # del loss_dict, losses
        # del loss_dict['zero']

        # ##########################################################################
        # #################### (3): train D with target domain #####################
        # #################################################################
        # loss_dict, features_lc_t, features_gl_t, score_maps_t = foward_detector(model, images_t, return_maps=True)
        #
        # for layer in used_feature_layers:
        #     # detatch score_map
        #     for map_type in score_maps_t[layer]:
        #         score_maps_t[layer][map_type] = score_maps_t[layer][map_type].detach()
        #
        #     if USE_DIS_DETECT_GL:
        #         loss_dict["loss_detect_%s_dt" % layer] = \
        #             dt_dis_lambda * model["d_dis_%s" % layer](features_lc_t[layer], target_label, domain='target')
        #
        #     if USE_DIS_GLOBAL:
        #         loss_dict["loss_adv_%s_dt" % layer] = \
        #             ga_dis_lambda * model["dis_%s" % layer](features_gl_t[layer], target_label, domain='target')
        #     if USE_DIS_CENTER_AWARE:
        #         loss_dict["loss_adv_%s_CA_dt" %layer] = \
        #             ca_dis_lambda * model["dis_%s_CA" % layer](features_gl_t[layer], target_label, score_maps_t[layer], domain='target')
        #     if USE_CM_GLOBAL:
        #         loss_dict["loss_cm_%s_dt" % layer] = \
        #             cm_dis_lambda * model["cm_%s" % layer](features_gl_t[layer], target_label, score_maps_t, None, layer, domain='target')
        #         #print("t_loss_cm_%s_ds" % layer, ':', loss_dict["loss_cm_%s_ds" % layer])
        #
        # losses = sum(loss for loss in loss_dict.values())
        #
        # # del "zero" (useless after backward)
        # del loss_dict['zero']

        # # reduce losses over all GPUs for logging purposes
        # if reduce_loss_dict:
        #     loss_dict_reduced = reduce_loss_dict(loss_dict)
        #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #     meters.update(loss_dt=losses_reduced, **loss_dict_reduced)
        #
        #     # saved GRL gradient
        #     grad_list = []
        #     for layer in used_feature_layers:
        #         def save_grl_grad(grad):
        #             grad_list.append(grad)
        #         features_lc_t[layer].register_hook(save_grl_grad)
        #
        #     losses.backward()
        #
        # # Uncomment to log GRL gradient
        # grl_grad = {}
        # grl_grad_log = {}
        #
        # del loss_dict, losses, grad_list, grl_grad, grl_grad_log

        ##########################################################################
        ##########################################################################
        ##########################################################################

        # optimizer.step()
        for k in optimizer:
            optimizer[k].step()

        if pytorch_1_1_0_or_later:
            # scheduler.step()
            for k in scheduler:
                scheduler[k].step()

        # End of training
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        sample_layer = used_feature_layers[0]  # sample any one of used feature layer

        sample_optimizer = optimizer["backbone"]
        if USE_DIS_DETECT_GL:
            sample_optimizer = optimizer["d_dis_%s" % sample_layer]
        if USE_DIS_GLOBAL:
            sample_optimizer = optimizer["dis_%s" % sample_layer]
        if USE_DIS_CENTER_AWARE:
            sample_optimizer = optimizer["dis_%s_CA" % sample_layer]
        if USE_DIS_BIN:
            sample_optimizer = optimizer["dis_%s_BIN" % sample_layer]
        

        if iteration % 20 == 0 or iteration == max_iter:
            
            logger.info(
                meters.delimiter.join([
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr_backbone: {lr_backbone:.6f}",
                    "lr_fcos: {lr_fcos:.6f}",
                    "lr_dis: {lr_dis:.6f}",
                    "max mem: {memory:.0f}",
                ]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                    lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                    lr_dis=sample_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ))

        root_path = os.getcwd()

        # if iteration % checkpoint_period == 0:
        if (iteration < 5000 and iteration % 60 == 0) or iteration % checkpoint_period == 0:
        # if 1:
            # if iteration % 20 == 0:
            #     checkpointer.save(os.path.join(arguments["model_save_path"], "model_{:07d}".format(iteration)), **arguments)
            #     checkpointer.save(os.path.join(root_path, arguments["model_save_path"], "model_{:07d}".format(iteration)), iteration, **arguments)
            # print('save_path: ', os.path.join(arguments["model_save_path"], "model_{:07d}".format(iteration)))
            # checkpointer.save(os.path.join(root_path, arguments["model_save_path"], "model_last"), iteration, **arguments)
            # # ###-----------------pp 增加评估，保存当前最好的模型-----------------
            # print('distributed: ', distributed)
            map_0_5 = run_test(cfg, copy.deepcopy(model), distributed)
            print('map_0_5: ', map_0_5)
            print('map_0_5_max_latest: ', map_0_5_max)
            if map_0_5 > map_0_5_max:
                map_0_5_max = map_0_5
                print('map_0_5_max: ', map_0_5_max)
                checkpointer.save(os.path.join(root_path, arguments["model_save_path"], "model_best"), iteration,
                                  map_0_5_max, **arguments)
            checkpointer.save(os.path.join(root_path, arguments["model_save_path"], "model_last"), iteration,
                              map_0_5_max, **arguments)

        # if iteration > 5000 and iteration <= 6000 and iteration % 50 == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save(root_path, os.path.join(arguments["model_save_path"], "model_final"), iteration,
                              map_0_5_max, **arguments)

        if iteration in [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            checkpointer.save(os.path.join(root_path, arguments["model_save_path"], "model_{:07d}".format(iteration)),
                              iteration, map_0_5_max, **arguments)

        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        # if iteration == max_iter:
        #     checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))

