shift
export CUDA_VISIBLE_DEVICES=3
python train.py \
--name CVACT \
--dataset CVACT \
--output_dir /raid/yanghongji/transformer/CVACT/ \
--learning_rate 1e-4 \
--weight_decay 0.03 \
--emb_size 384 \
--img_grd_size 128 512 \
--img_sat_size 128 512 \
--polar 1 \
--loss_weight 10 \
--model_type SAIG_S \
--pretrained_dir /raid/yanghongji/transformer/SAIG_S.pth.tar \
--dataset_dir /raid/luxiufan/CVACT/ANU_DATA_SMALL/ANU_data_small/