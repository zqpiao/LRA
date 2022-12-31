# Unsupervised Domain-Adaptive Object Detection via Localization Regression Alignment---Faster R-CNN


## Installation 

The implementation of our anchor-based detector is heavily based on Faster-RCNN ([\#f0a9731](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption)).


## Dataset

refer to  SAPNet(https://isrc.iscas.ac.cn/gitlab/research/domain-adaption) for all details of dataset construction.


## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.


**1. Pre-training with only GA module**


- Using VGG-16 as backbone with 2 GPUs

export CUDA_VISIBLE_DEVICES=0,1


 ```
python-m torch.distributed.launch --nproc_per_node=2 --master_port=2900 dis_train.py --config-file configs/mgad/cityscape_to_foggy/VGG/s0/adv_vgg16_cityscapes_2_foggy.yaml
```




## Evaluation

The trained model can be evaluated by the following command.

```
python-m torch.distributed.launch --nproc_per_node=2 --master_port=2900 dis_train.py --config-file configs/mgad/cityscape_to_foggy/VGG/s1/adv_vgg16_cityscapes_2_foggy.yaml --resume /your_path/ --test-only
```

**Environments**

- Hardware
  - 2 NVIDIA A100 GPUs

- Software
  - Python 3.8.5
  - PyTorch 1.7.1
  - Torchvision 0.8.2
  - CUDA 11.1



## Citations


