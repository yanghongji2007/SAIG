shift
export CUDA_VISIBLE_DEVICES=3,4
python train_vigor_sam.py \
--name VIGOR \
--dataset VIGOR \
--learning_rate 1e-4 \
--weight_decay 0.03 \
--model_type SAIG_S \
--pool GAP \
--emb_size 384 \
--total_epoch 90 \
--max_grad_norm 1.0 \
--pretrained_dir /raid/yanghongji/transformer/SAIG_S.pth.tar \
--dataset_dir /raid/yanghongji/VIGOR/ \
--output_dir /raid/yanghongji/transformer/output_vigor_SAIG_S_SAM_GAP/ \
