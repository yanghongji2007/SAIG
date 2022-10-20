#!/bin/bash
shift
export CUDA_VISIBLE_DEVICES=4,5
python3 -m torch.distributed.launch --master_port 66666 --nproc_per_node=2 train.py \
--data_dir /raid/zhuyingying/imagenet-1k/ \
--num-classes 1000 \
--img-size 224 \
--opt adamw \
--weight-decay 0.01 \
--lr .001 \
--epochs 300 \
--output /data/zhuyingying/swin_transformer/pytorch-image-models-master/output/ \
--sched cosine \
--aa rand-m9-mstd0.5-inc1 \
--remode pixel \
--model vit_small_patch16_224 \
-j 8 \
-b 256 \
--mixup .8 \
--cutmix 1.0 \
--warmup-epochs 5 \
--reprob 0.25 \
--drop-path 0.1 \
--amp \
--warmup-lr 1e-6