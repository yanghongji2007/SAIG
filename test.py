import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io as scio
import torch.nn.functional as F
import argparse

from timm.models.SAIG import SAIG_Deep, SAIG_Shallow, resize_pos_embed

from model.model import twoviewmodel
import time
import random
import math
import torchvision


def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i,i]
        prediction = torch.sum(dist_array[:, i] < gt_dist) 
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring.")

parser.add_argument("--dataset", choices=["CVUSA", "CVACT"], default="CVUSA",
                        help="Which downstream task.")

parser.add_argument("--model_type", choices=["SAIG_D", "SAIG_S"],
                        default="SAIG_D",
                        help="Which variant to use.")

parser.add_argument("--pool", choices=["GAP", "SMD"],
                        default="GAP",
                        help="Which pooling layer to use.")

parser.add_argument("--polar", type=int,choices=[1,0],
                        default=1,
                        help="polar transform or not")

parser.add_argument("--dataset_dir", default="output", type=str,
                    help="The dataset path.")

parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

parser.add_argument("--img_grd_size", nargs='+', default=(128, 512), type=int,
                        help="Resolution size")

parser.add_argument("--img_sat_size", nargs='+', default=(256, 256), type=int,
                        help="Resolution size")

parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

parser.add_argument("--emb_size", default=384, type=int,
                        help="embedding size")

args = parser.parse_args()

if args.model_type == 'SAIG_D':
    model_grd = SAIG_Deep(img_size = args.img_grd_size)
    model_sat = SAIG_Deep(img_size = args.img_sat_size)
elif args.model_type == 'SAIG_S':
    model_grd = SAIG_Shallow(img_size = args.img_grd_size)
    model_sat = SAIG_Shallow(img_size = args.img_sat_size)

model_grd.reset_classifier(0)
model_sat.reset_classifier(0)

model = twoviewmodel(model_grd, model_sat, args)

#model = nn.DataParallel(model)

print("loading model form ", os.path.join(args.output_dir,'model_checkpoint.pth'))

state_dict = torch.load(os.path.join(args.output_dir,'model_checkpoint.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict['model'])



if args.dataset == 'CVUSA':
    from utils.dataloader_usa import TestDataloader
elif args.dataset == 'CVACT':
    from utils.dataloader_act import TestDataloader

testset = TestDataloader(args)
test_loader = DataLoader(testset,
                        batch_size=args.eval_batch_size,
                        shuffle=False, 
                        num_workers=4)

model.cuda()

sat_global_descriptor = torch.zeros([8884, args.emb_size]).cuda()
grd_global_descriptor = torch.zeros([8884, args.emb_size]).cuda()
"""
sat_global_descriptor = torch.zeros([8884, 3072])#.cuda()
grd_global_descriptor = torch.zeros([8884, 3072])#.cuda()
"""
val_i =0

model.eval()

with torch.no_grad():
    for step, (x_grd, x_sat) in enumerate(tqdm(test_loader)):  
        x_grd, x_sat = x_grd.cuda(), x_sat.cuda()
        with torch.cuda.amp.autocast():
            grd_global,sat_global = model(x_grd,x_sat)

        sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.cpu().detach()
        grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.cpu().detach()
        val_i += sat_global.shape[0]


#scio.savemat('./sat_global_descriptor.mat', {'sat_global_descriptor':sat_global_descriptor.cpu().numpy()})
#scio.savemat('./grd_global_descriptor.mat', {'grd_global_descriptor':grd_global_descriptor.cpu().numpy()})

print('   compute accuracy')
dist_array = 2.0 - 2.0 * torch.matmul(sat_global_descriptor, grd_global_descriptor.T)

top1_percent = int(dist_array.shape[0] * 0.01) + 1
val_accuracy = torch.zeros((1, top1_percent)).cuda()

print('start')

print('top1', ':', validate(dist_array, 1))
print('top5', ':', validate(dist_array, 5))
print('top10', ':', validate(dist_array, 10))
print('top1%', ':', validate(dist_array, top1_percent))


