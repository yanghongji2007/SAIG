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


class SpatialAware(nn.Module):
    def __init__(self, in_channel, d=8):
        super().__init__()
        c = in_channel
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c // 2),
                nn.Linear(c // 2, c),
            ) for _ in range(d)
        ])

    def forward(self, x):
        # channel dim
        x = torch.mean(x, dim=1)
        x = [b(x) for b in self.fc]
        x = torch.stack(x, -1)
        return x

class SAFABranch(nn.Module):
    def __init__(self,model,N):
        super().__init__()
        self.backbone = model
        self.pooling = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        print(N)
        self.spatial_aware = SpatialAware(N, d=8)
        
    def forward(self, x):
        
        H, W = x.shape[2]//16, x.shape[3]//16
        x = self.backbone(x) # [B, 256, 384]

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # [B, 384, 8, 32]
        x = self.pooling(x) # [B, 384, 4, 16]

        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x_sa = self.spatial_aware(x)
        # b c h*w @ b h*w d = b c d
        x = x @ x_sa
        x = torch.transpose(x, -1, -2).flatten(-2, -1) # [B, 8*384]
        return x

class MixerHead_Branch(nn.Module):
    def __init__(self, model,N):
        super().__init__()
        self.backbone = model
        """
        self.linear = nn.Linear(N, 8)
        """
        self.linear1 = nn.Linear(N, N*4)
        #self.linear1 = nn.Linear(256, 1024)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(N*4, N)
        self.linear3 = nn.Linear(N, 8)
        
    def forward(self, x):
        
        H, W = x.shape[2]//16, x.shape[3]//16

        x = self.backbone(x)

        x = x.transpose(-1,-2)
        """
        x=self.linear(x)
        """
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        x = self.linear3(x)
        
        x = x.transpose(-1,-2) # [B, 8, 384]
        #x = torch.flatten(x, start_dim=-3, end_dim=-1) # [B, 48*4*16]
        x = x.flatten(-2,-1)

        return x


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring.")

parser.add_argument("--dataset", choices=["VIGOR"], default="VIGOR",
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

parser.add_argument("--emb_size", default=3072, type=int,
                        help="embedding size")    
parser.add_argument("--img_grd_size", nargs='+', default=(320, 640), type=int,
                        help="Resolution size of ground image")

parser.add_argument("--img_sat_size", nargs='+', default=(320, 320), type=int,
                        help="Resolution size of satellite image")

parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

args = parser.parse_args()

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device
"""



if args.model_type == 'SAIG_D':
    model_grd = SAIG_Deep(img_size = args.img_grd_size)
    model_sat = SAIG_Deep(img_size = args.img_sat_size)
elif args.model_type == 'SAIG_S':
    model_grd = SAIG_Shallow(img_size = args.img_grd_size)
    model_sat = SAIG_Shallow(img_size = args.img_sat_size)
    
model_grd.reset_classifier(0)
model_sat.reset_classifier(0)

if args.pool == 'GAP':
    model_grd = model_grd
    model_sat = model_sat
elif args.pool == 'SMD':
    model_grd = MixerHead_Branch(model_grd, args.img_grd_size[0]//16*args.img_grd_size[1]//16)
    model_sat = MixerHead_Branch(model_sat, args.img_sat_size[0]//16*args.img_sat_size[1]//16)

model_grd = nn.DataParallel(model_grd)
model_sat = nn.DataParallel(model_sat)

print("loading model form ", os.path.join(args.output_dir,'model_grd_checkpoint.pth'))

state_dict = torch.load(os.path.join(args.output_dir,'model_checkpoint.pth'))
model_grd.load_state_dict(state_dict['model_grd'])
model_sat.load_state_dict(state_dict['model_sat'])



if args.dataset == 'CVUSA':
    from utils.dataloader_usa import TestDataloader
elif args.dataset == 'CVACT':
    from utils.dataloader_act import TestDataloader
elif args.dataset == 'VIGOR':
    from utils.dataloader_VIGOR import TestDataloader_sat, TestDataloader_grd

testset_sat = TestDataloader_sat(args)
test_loader_sat = DataLoader(testset_sat,
                        batch_size=args.eval_batch_size,
                        shuffle=False, 
                        num_workers=8)

testset_grd = TestDataloader_grd(args)
test_loader_grd = DataLoader(testset_grd,
                        batch_size=args.eval_batch_size,
                        shuffle=False, 
                        num_workers=8)

print(test_loader_grd.dataset.test_label.shape)
model_grd.cuda()
model_sat.cuda()

sat_global_descriptor = torch.zeros([test_loader_sat.dataset.test_sat_data_size, args.emb_size]).cuda()
grd_global_descriptor = torch.zeros([test_loader_grd.dataset.test_data_size, args.emb_size]).cuda()


model_grd.eval()
model_sat.eval()


with torch.no_grad():
    val_i =0

    for step, x_grd in enumerate(tqdm(test_loader_grd)):
        x_grd = x_grd.cuda()
        with torch.cuda.amp.autocast():
            grd_global = model_grd(x_grd)
            if args.pool == 'GAP':
                grd_global = nn.AdaptiveAvgPool1d(1)(grd_global.transpose(-1, -2)).squeeze(2)
            grd_global = F.normalize(grd_global, dim=1)
        grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach()

        val_i += grd_global.shape[0]
    val_i =0

    for step, x_sat in enumerate(tqdm(test_loader_sat)):
        x_sat = x_sat.cuda()
        with torch.cuda.amp.autocast():
            sat_global = model_sat(x_sat)
            if args.pool == 'GAP':
                sat_global = nn.AdaptiveAvgPool1d(1)(sat_global.transpose(-1, -2)).squeeze(2)
            sat_global = F.normalize(sat_global, dim=1)
        sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach()

        val_i += sat_global.shape[0]


# scio.savemat('./vigor_sat_global_descriptor.mat', {'sat_global_descriptor':sat_global_descriptor.cpu().numpy()})
# scio.savemat('./vigor_grd_global_descriptor.mat', {'grd_global_descriptor':grd_global_descriptor.cpu().numpy()})

print('   compute accuracy')
grd_global_descriptor = grd_global_descriptor.cpu().numpy()
sat_global_descriptor = sat_global_descriptor.cpu().numpy()
    
accuracy = 0.0
accuracy_top1 = 0.0
accuracy_top5 = 0.0
accuracy_top10 = 0.0
accuracy_top100 = 0.0
accuracy_hit = 0.0

data_amount = 0.0
    
dist_array = 2 - 2 * np.matmul(grd_global_descriptor, sat_global_descriptor.T)

print(dist_array.shape)

top1_percent = int(dist_array.shape[1] * 0.01) + 1
top1 = 1
top5 = 5
top10 = 10
top100 = 100

for i in range(dist_array.shape[0]):
    gt_dist = dist_array[i, test_loader_grd.dataset.test_label[i][0]] # positive sat

    prediction = np.sum(dist_array[i, :] < gt_dist)

    dist_temp = np.ones(dist_array[i, :].shape[0])
    
    dist_temp[test_loader_grd.dataset.test_label[i][1:]] = 0 # cover semi-positive sat

    prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

    if prediction < top1_percent:
        accuracy += 1.0
    if prediction < top1:
        accuracy_top1 += 1.0
    if prediction < top5:
        accuracy_top5 += 1.0
    if prediction < top10:
        accuracy_top10 += 1.0
    if prediction < top100:
        accuracy_top100 += 1.0
    if prediction_hit < top1:
        accuracy_hit += 1.0
    data_amount += 1.0

accuracy /= data_amount
accuracy_top1 /= data_amount
accuracy_top5 /= data_amount
accuracy_top10 /= data_amount
accuracy_top100 /= data_amount
accuracy_hit /= data_amount

print('accuracy = %.2f%% , top1: %.2f%%, top5: %.2f%%, top10: %.2f%%, top100: %.2f%%,hit_rate: %.2f%%' % (
            accuracy * 100.0, accuracy_top1 * 100.0, accuracy_top5 * 100.0, accuracy_top10 * 100.0, accuracy_top100 * 100.0, accuracy_hit * 100.0))