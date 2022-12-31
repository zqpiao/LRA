# Unsupervised Domain-Adaptive Object Detection via Localization Regression Alignment---FCOS


## Installation 

Check [INSTALL.md] for installation instructions. 

The implementation of our anchor-free detector is heavily based on FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).


## Dataset

refer to  EveryPixelMatters(https://github.com/chengchunhsu/EveryPixelMatters) for all details of dataset construction.


## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.


**1. Pre-training with only GA module**

export PYTHONPATH=$PWD:$PYTHONPATH

- Using VGG-16 as backbone with 2 GPUs

export CUDA_VISIBLE_DEVICES=0,1

[first stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_mgad.py --config-file ./configs/detector/cityscapes_to_foggy/VGG/S0/bin_sup_only_da_ga_cityscapes_VGG_16_FPN_4x.yaml OUTPUT_DIR /your_path/ MODEL.ISSAMPLE True
```

[second stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_mgad.py --config-file ./configs/detector/cityscapes_to_foggy/VGG/S1/bin_da_ga_cityscapes_VGG_16_FPN_4x_from_s0_bin_sup.yaml OUTPUT_DIR ./ MODEL.ISSAMPLE True
```


## Evaluation

The trained model can be evaluated by the following command.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2300 tools/test_net_mgad.py --config-file configs/detector/cityscapes_to_foggy/VGG/S1/da_ga_cityscapes_VGG_16_FPN_4x.yaml MODEL.WEIGHT ./model_rs.pth
```

**Environments**

- Hardware
  - 2 NVIDIA A100 GPUs

- Software
  - Python 3.8.8
  - PyTorch 1.7.1
  - Torchvision 0.2.1
  - CUDA 11.1



## Citations



