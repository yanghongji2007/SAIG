shift
export CUDA_VISIBLE_DEVICES=6
python test.py \
--name CVACT \
--dataset CVACT \
--output_dir /raid/yanghongji/transformer/CVACT/nopolar/SAIG-D/ \
--emb_size 3072 \
--model_type SAIG_D \
--pool SMD \
--img_grd_size 128 512 \
--img_sat_size 256 256 \
--polar 0 \
--dataset_dir /raid/luxiufan/CVACT/ANU_DATA_SMALL/ANU_data_small/