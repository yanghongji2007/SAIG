shift
export CUDA_VISIBLE_DEVICES=6,7
python train_VIGOR.py \
--name VIGOR \
--dataset VIGOR \
--learning_rate 1e-4 \
--weight_decay 0.03 \
--model_type SAIG_S \
--total_epoch 90 \
--max_grad_norm 1.0 \
--pretrained_dir /raid/yanghongji/transformer/SAIG_S.pth.tar \
--dataset_dir /raid/yanghongji/VIGOR/ \
--output_dir /raid/yanghongji/transformer/output_vigor/ \
