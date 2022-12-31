import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import pycocotools.mask as mask_util

from detection import utils
from detection.data.evaluations import coco_evaluation, voc_evaluation
from detection.data.transforms import de_normalize
from detection.layers.mask_ops import paste_masks_in_image
from detection.utils import colormap
from detection.utils.dist_utils import is_main_process, all_gather, get_world_size
from detection.utils.visualizer import Visualizer

from .timer import Timer, get_time_str


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def evaluation(model, data_loaders, device, types=('coco',), output_dir='./evaluations/', iteration=None, viz=False):
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = (data_loaders,)
    results = dict()
    for data_loader in data_loaders:
        dataset = data_loader.dataset
        _output_dir = os.path.join(output_dir, 'evaluations', dataset.dataset_name)
        os.makedirs(_output_dir, exist_ok=True)
        result = do_evaluation(model, data_loader, device, types=types, output_dir=_output_dir, iteration=iteration, viz=viz)
        results[dataset.dataset_name] = result
    return results


COLORMAP = colormap.colormap(rgb=True, maximum=1)


def save_visualization(dataset, img_meta, result, output_dir, threshold=0.8, fmt='.pdf'):
    save_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    file_name = img_meta['img_info']['file_name']
    img = Image.open(os.path.join(dataset.images_dir, file_name))
    w, h = img.size
    scale = 1.0
    # w, h = int(w * scale), int(h * scale)
    # img = img.resize((w, h))

    vis = Visualizer(img, metadata=None)

    boxes = np.array(result['boxes']) * scale
    labels = np.array(result['labels'])
    scores = np.array(result['scores'])
    #masks = result['masks']
    indices = scores > threshold
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    #masks = [m for m, b in zip(masks, indices) if b]

    # colors = COLORMAP[(labels + 2) % len(COLORMAP)]

    colors = [np.array([1.0, 0, 0])] * len(labels)

    labels = ['{}:{:.0f}%'.format(dataset.CLASSES[label], score * 100) for label, score in zip(labels, scores)]
    out = vis.overlay_instances(
        boxes=boxes,
        labels=labels,
        masks=None,
        assigned_colors=colors,
        alpha=0.8,
    )
    out.save(os.path.join(save_dir, os.path.basename(file_name).replace('.', '_') + fmt))


import cv2 as cv
import random
random.seed(11)


def NMS(dets, scores, thresh):

    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]


    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序
    # print('scores: ', scores.shape)
    # print('order: ', order)


    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    #
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv.rectangle(img, c1, c2, color, thickness=tl)
    cv.rectangle(img, c1, c2, color, thickness=4)
    # cv.circle(img, c1, 3, color, thickness = 8)

    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]) + 5)
    # # cv.rectangle(img, c1, c2, color, thickness=tl)
    # cv.rectangle(img, c1, c2, color, thickness=1)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1)  # filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)



@torch.no_grad()
def do_evaluation(model, data_loader, device, types, output_dir, iteration=None, viz=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    dataset = data_loader.dataset
    header = 'Testing {}:'.format(dataset.dataset_name)
    results_dict = {}
    has_mask = False

    show_class_names = ["person", "car", "train", "rider", "truck", "motorcycle", "bicycle", "bus"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(show_class_names))]
    save_path = r"/data/piaozhengquan/projects/DAOD/MGADA-master/Faster-RCNN/debug/results_show/ours_show_city2fog"
    # save_path = r"/data/piaozhengquan/projects/DAOD/MGADA-master/Faster-RCNN/debug/results_show/MGA_show_city2fog"

    score_th = 0.3
    save_path = save_path + str(score_th)


    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            os.mkdir(save_path)

    inference_timer = Timer()
    for images, img_metas, targets in metric_logger.log_every(data_loader, 10, header):
        assert len(targets) == 1
        images = images.to(device)
        inference_timer.tic()
        #print(images.size())

        model_time = time.time()
        det = model(images, img_metas)[0]
        boxes, scores, labels = det['boxes'], det['scores'], det['labels']

        model_time = time.time() - model_time

        img_meta = img_metas[0]
        scale_factor = img_meta['scale_factor']
        img_info = img_meta['img_info']



        ####保存可视化结果
        img_name = img_info['file_name']
        # img_path = os.path.join('/data/piaozhengquan/datasets/Cityscapes/leftImg8bit_foggy/val', img_name)
        # img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8),
        #                   cv.IMREAD_UNCHANGED)  # 打开含有中文路径的图片

        img = de_normalize(images[0], img_meta)
        img = img[:, :, [2, 1, 0]]
        img = np.ascontiguousarray(img)

        for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.tolist(), labels.tolist())):
            # if scores[i] > 0.65:
            if scores[i] > score_th:
                xyxy = (x1, y1, x2, y2)
                category_id = dataset.label2cat[label]
                plot_one_box(xyxy, img, color=colors[int(category_id) - 1])

        cv.imencode('.jpg', img)[1].tofile(os.path.join(save_path, img_name.split('/')[-1].split('.')[0]+'.jpg'))


        if viz:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.switch_backend('TKAgg')
            image = de_normalize(images[0], img_meta)
            plt.subplot(122)
            plt.imshow(image)
            plt.title('Predict')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.tolist(), labels.tolist())):
                if scores[i] > 0.65:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                    category_id = dataset.label2cat[label]
                    plt.text(x1, y1, '{}:{:.2f}'.format(dataset.CLASSES[category_id], scores[i]), color='r')
                    plt.gca().add_patch(rect)

            plt.subplot(121)
            plt.imshow(image)
            plt.title('GT')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(targets[0]['boxes'].tolist(), targets[0]['labels'].tolist())):
                category_id = dataset.label2cat[label]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                plt.text(x1, y1, '{}'.format(dataset.CLASSES[category_id]))
                plt.gca().add_patch(rect)
            plt.show()

        boxes /= scale_factor
        result = {}

        if 'masks' in det:
            has_mask = True
            (w, h) = img_meta['origin_img_shape']
            masks = paste_masks_in_image(det['masks'], boxes, (h, w))
            rles = []
            for mask in masks.cpu().numpy():
                mask = mask >= 0.5
                mask = mask_util.encode(np.array(mask[0][:, :, None], order='F', dtype='uint8'))[0]
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                mask['counts'] = mask['counts'].decode('utf-8')
                rles.append(mask)
            result['masks'] = rles

        boxes = boxes.tolist()
        labels = labels.tolist()
        labels = [dataset.label2cat[label] for label in labels]
        scores = scores.tolist()

        result['boxes'] = boxes
        result['scores'] = scores
        result['labels'] = labels

        #save_visualization(dataset, img_meta, result, output_dir, fmt='.jpg')

        results_dict.update({
            img_info['id']: result
        })
        metric_logger.update(model_time=model_time)
        inference_timer.toc()

    if get_world_size() > 1:
        dist.barrier()

    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    if not is_main_process():
        return {}
    results = {}
    if has_mask:
        result = coco_evaluation(dataset, predictions, output_dir, iteration=iteration)
        results.update(result)
    if 'voc' in types:
        result = voc_evaluation(dataset, predictions, output_dir, iteration=iteration, use_07_metric=False)
        results.update(result)
        
    print(results)
        
    print("fps:", str(inference_timer.total_time / 50))
    return results
