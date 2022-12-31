#!/usr/bin/env bash
#IMAGENET_PRETRAIN=/home/piaozhengquan/Desktop/data/piaozhengquan/projects/FSOD/DeFRCN-main/pretrain_weights/MSRA/R-101.pkl                            # <-- change it to you path
#IMAGENET_PRETRAIN_TORCH=/home/piaozhengquan/Desktop/data/piaozhengquan/projects/FSOD/DeFRCN-main/pretrain_weights/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path



##递归遍历
#traverse_dir()
#{
#    filepath=$1
#
#    for file in `ls -a $filepath`
#    do
#        if [ -d ${filepath}/$file ]
#        then
#            if [[ $file != '.' && $file != '..' ]]
#            then
#                #递归
#                traverse_dir ${filepath}/$file
#            fi
#        else
#            #调用查找指定后缀文件
#            check_suffix ${filepath}/$file
#        fi
#    done
#}
###获取后缀为txt或ini的文件
#check_suffix()
#{
#    file=$1
#
#    if [ "${file##*.}"x = "pth"x ];then
#        echo $file
#    fi
#}

traverse_dir()
{
    filepath=$1


}




#model_path="/data/piaozhengquan/projects/DAOD/MGADA-master/ori_FCOS/debug/S1_da_ga_cityscapes_VGG_16_FPN_4x_bs_14"
model_path="/data/piaozhengquan/projects/DAOD/MGADA-master/ori_FCOS/debug/S1_da_ga_sim10k_VGG_16_FPN_4x_bs_14"



filepath=$model_path
for file in `ls -a $filepath`
    do

        file=${filepath}/$file
        if [ "${file##*.}"x = "pth"x ];then
        echo $file

        ##python test.py
#        python -m torch.distributed.launch --nproc_per_node=2 --master_port=2300 tools/test_net_mgad.py --config-file configs/detector/cityscapes_to_foggy/VGG/S1/da_ga_cityscapes_VGG_16_FPN_4x.yaml MODEL.WEIGHT $file

        python -m torch.distributed.launch --nproc_per_node=2 --master_port=2300 tools/test_net_mgad.py --config-file configs/detector/sim10k_to_cityscape/VGG/S1/adv_vgg16_sim10k_2_cityscape.yaml MODEL.WEIGHT $file


        fi
    done


#GPUS=2
#PORT=${PORT:-49501}



#SPLIT_ID=1
##SPLIT_ID=3
#for seed in 1 2 3 4 5 6 7 8 9
##for seed in 1
#do
#    for shot in 10 30  # if final, 10 -> 1 2 3 5 10
##    for shot in 1
##    for shot in 1 2   # if final, 10 -> 1 2 3 5 10
#    do
##        python $(dirname "$0")/create_config.py --dataset voc --config_root=${CONFIG_ROOT}          \
##            --shot ${shot} --seed ${seed} --setting ''
#
#
##        CONFIG_PATH=${CONFIG_ROOT}/split${SPLIT_ID}/fsce_r101_fpn_${setting}voc-split${SPLIT_ID}_${shot}shot-fine-tuning_seed${seed}_energy_discri.py
##        CONFIG_PATH=work_dir_new/coco/seperate/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri.py
##        python $(dirname "$0")/coco_test.py --config=$CONFIG_PATH --work-dir=work_dir_new/coco/seperate/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri
#
#        CONFIG_PATH=work_dir_new_coco/coco/seperate/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri_energy2/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri_energy_2.py
#        python $(dirname "$0")/coco_test.py --config=$CONFIG_PATH --work-dir=work_dir_new_coco/coco/seperate/fsce_r101_fpn_coco_${shot}shot-fine-tuning_seed${seed}_discri_energy2
#
#        sleep 10
#
#
#    done
#done
