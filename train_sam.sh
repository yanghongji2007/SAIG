shift
export CUDA_VISIBLE_DEVICES=5
python train_sam.py \
--name CVACT \
--dataset CVACT \
--output_dir /raid/yanghongji/transformer/output_act/ \
--learning_rate 0.0001 \
--weight_decay 0.03 \
--img_grd_size 128 512 \
--img_sat_size 256 256 \
--polar 0 \
--loss_weight 10 \
--model_type SAIG_D \
--pool GAP \
--emb_size 384 \
--pretrained_dir /raid/yanghongji/transformer/SAIG_D.pth.tar \
--dataset_dir /raid/luxiufan/CVACT/ANU_DATA_SMALL/ANU_data_small/