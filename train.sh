shift
export CUDA_VISIBLE_DEVICES=6
python train.py \
--name CVUSA \
--dataset CVUSA \
--output_dir /raid/yanghongji/transformer/output_usa/SAIGD_polar_test/ \
--learning_rate 0.0001 \
--weight_decay 0.03 \
--emb_size 384 \
--img_grd_size 128 512 \
--img_sat_size 256 256 \
--polar 0 \
--model_type SAIG_D \
--pretrained_dir /raid/yanghongji/transformer/SAIG_D.pth.tar \
--dataset_dir /raid/luxiufan/Polar_CVUSA/ 