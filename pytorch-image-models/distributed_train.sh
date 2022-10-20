#!/bin/bash
shift
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m torch.distributed.launch --master_port 66669 --nproc_per_node=4 train.py \
--data_dir /raid/yanghongji/imagenet-21k/ImageNet2012/ \
--num-classes 1000 \
--img-size 224 \
--opt adamw \
--weight-decay 0.01 \
--lr .001 \
--epochs 300 \
--output /raid/yanghongji/cross_view_localization/pytorch-image-models/output/ \
--sched cosine \
--aa rand-m9-mstd0.5-inc1 \
--remode pixel \
--model vit_small_patch16_224 \
-j 8 \
-b 128 \
--mixup .8 \
--cutmix 1.0 \
--warmup-epochs 5 \
--reprob 0.25 \
--drop-path 0.1 \
--amp \
--model-ema \
--warmup-lr 1e-6