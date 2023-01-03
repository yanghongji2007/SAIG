# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from torch.nn import functional as F
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from timm.models.SAIG import SAIG_Deep, SAIG_Shallow, resize_pos_embed

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, ConstantLRSchedule
from utils.data_utils import get_loader
from utils.loss import InfoNCE, triplet_loss, SemiSoftTriHard
import math
import itertools
from utils.sam import SAM
import torchvision


logger = logging.getLogger(__name__)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_model(args, model_grd, model_sat,optimizer,epoch,best_acc):
    
    model_checkpoint = os.path.join(args.output_dir, "model_checkpoint.pth")
    checkpoint = {
        'model_grd':model_grd.state_dict(),
        'model_sat':model_sat.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':epoch,
        'best_acc':best_acc
    }
    torch.save(checkpoint, model_checkpoint)
    
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

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



def setup(args):
    # Prepare model
    if args.model_type == 'SAIG_D':
        model_grd = SAIG_Deep(img_size = args.img_grd_size)
        model_sat = SAIG_Deep(img_size = args.img_sat_size)
    elif args.model_type == 'SAIG_S':
        model_grd = SAIG_Shallow(img_size = args.img_grd_size)
        model_sat = SAIG_Shallow(img_size = args.img_sat_size)
    

    model_grd.reset_classifier(0)
    model_sat.reset_classifier(0)
    
    state_dict = torch.load(args.pretrained_dir)
    temp = state_dict['state_dict']['pos_embed']
    state_dict['state_dict']['pos_embed'] = resize_pos_embed(temp, model_grd.pos_embed,gs_new = (args.img_grd_size[0]//16, args.img_grd_size[1]//16))
    model_grd.load_state_dict(state_dict['state_dict'],strict=False)

    state_dict['state_dict']['pos_embed'] = resize_pos_embed(temp, model_sat.pos_embed,gs_new = (args.img_sat_size[0]//16, args.img_sat_size[1]//16))
    model_sat.load_state_dict(state_dict['state_dict'],strict=False)
    
    #model_grd = SAFABranch(model_grd, args.img_grd_size[0]//16*args.img_grd_size[1]//16//4)
    #model_sat = SAFABranch(model_sat, args.img_sat_size[0]//16*args.img_sat_size[1]//16//4)
    if args.pool == 'GAP':
        model_grd = model_grd
        model_sat = model_sat
    elif args.pool == 'SMD':
        model_grd = MixerHead_Branch(model_grd, args.img_grd_size[0]//16*args.img_grd_size[1]//16)
        model_sat = MixerHead_Branch(model_sat, args.img_sat_size[0]//16*args.img_sat_size[1]//16)

    print(model_grd)
    if torch.cuda.device_count() >1:
        print("Using ", torch.cuda.device_count(),"GPUs!")
        model_grd = nn.DataParallel(model_grd)
        model_sat = nn.DataParallel(model_sat)
        
    model_grd.cuda()
    model_sat.cuda()

    num_params = count_parameters(model_grd) + count_parameters(model_sat)


    logger.info("Training parameters %s", args)

    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model_grd, model_sat


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def valid(args, model_grd, model_sat, writer, test_loader_grd, test_loader_sat, epoch):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader_grd))
    logger.info("  Num steps = %d", len(test_loader_sat))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_grd.eval()
    model_sat.eval()
    
    # grd: 52605 sat: 90618
    sat_global_descriptor = torch.zeros([test_loader_sat.dataset.test_sat_data_size, args.emb_size]).cuda()
    grd_global_descriptor = torch.zeros([test_loader_grd.dataset.test_data_size, args.emb_size]).cuda()
    

    
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



    print('   compute accuracy')
    grd_global_descriptor = grd_global_descriptor.cpu().numpy()
    sat_global_descriptor = sat_global_descriptor.cpu().numpy()
    
    val_accuracy, val_accuracy_top1, val_accuracy_top5, hit_rate = validate_top(grd_global_descriptor,
                                                                              sat_global_descriptor, test_loader_grd)

    
    
    print('Evaluation epoch %d: accuracy = %.2f%% , top1: %.2f%%, top5: %.2f%%, hit_rate: %.2f%%' % (
            epoch, val_accuracy * 100.0, val_accuracy_top1 * 100.0, val_accuracy_top5 * 100.0, hit_rate * 100.0))
    # save eval result
    file = './Result/'+ args.dataset + '/' + str(args.model_type) + '_accuracy.txt'
    if not os.path.exists('./Result/'+ args.dataset):
        os.makedirs('./Result/'+ args.dataset)
    with open(file, 'a') as file:
        file.write(str(epoch) + ' ' + ' : ' + str(val_accuracy_top1*100.0) + '  '+ str(val_accuracy_top5*100.0) + '\n')

    # print the valid information
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Accuracy: %2.5f" % val_accuracy_top1)

    writer.add_scalar("test/accuracy", scalar_value=val_accuracy_top1, global_step=epoch)

    return val_accuracy, val_accuracy_top1, val_accuracy_top5, hit_rate

def validate_top(grd_descriptor, sat_descriptor, dataloader):
    accuracy = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    
    print('start')

    dist_array = 2 - 2 * np.matmul(grd_descriptor, sat_descriptor.T)

    print(dist_array.shape)

    top1_percent = int(dist_array.shape[1] * 0.01) + 1
    top1 = 1
    top5 = 5
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, dataloader.dataset.test_label[i][0]]

        prediction = np.sum(dist_array[i, :] < gt_dist)

        dist_temp = np.ones(dist_array[i, :].shape[0])
        dist_temp[dataloader.dataset.test_label[i][1:]] = 0
        prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

        if prediction < top1_percent:
            accuracy += 1.0
        if prediction < top1:
            accuracy_top1 += 1.0
        if prediction < top5:
            accuracy_top5 += 1.0
        if prediction_hit < top1:
            accuracy_hit += 1.0
        data_amount += 1.0
 
    accuracy /= data_amount
    accuracy_top1 /= data_amount
    accuracy_top5 /= data_amount
    accuracy_hit /= data_amount
    return accuracy, accuracy_top1, accuracy_top5, accuracy_hit

def train(args, model_grd, model_sat):
    """ Train the model """
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))


    # Prepare dataset
    train_loader, test_loader_grd, test_loader_sat= get_loader(args)

    
    # Prepare optimizer and scheduler
    base_opt = torch.optim.AdamW
    optimizer = SAM(itertools.chain(model_grd.parameters(), model_sat.parameters()), base_opt, 2, True, lr=args.learning_rate, weight_decay=args.weight_decay)

    print(optimizer)
    if args.resume:
        print(args.resume)
        state_dict = torch.load(os.path.join(args.resume,'model_checkpoint.pth'))
        model_grd.load_state_dict(state_dict['model_grd'])
        model_sat.load_state_dict(state_dict['model_sat'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        global_step = (start_epoch+1) * len(train_loader)
        best_acc = state_dict['best_acc']

        print('model start from ' + str(start_epoch) + ' epoch')
        print('best acc: ' + str(best_acc))

    else:


        start_epoch = args.start_epoch
        global_step, best_acc = 0, 0
    t_total = args.total_epoch * len(train_loader)
    

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # loss function
    #criterion = SemiSoftTriHard()
    criterion = triplet_loss()
    #criterion = InfoNCE()
    
    model_grd.zero_grad()
    model_sat.zero_grad()

    losses = AverageMeter()
    

    #scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.total_epoch):
        model_grd.train()
        model_sat.train()

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                             )



        for step, (x_grd, x_sat) in enumerate(epoch_iterator):
            x_grd, x_sat=x_grd.cuda(), x_sat.cuda()
            
            #optimizer.zero_grad()
            #enable_running_stats(model_grd)
            #enable_running_stats(model_sat)
            #with torch.cuda.amp.autocast():
                
            grd_global = model_grd(x_grd)
            sat_global = model_sat(x_sat)
            if args.pool == 'GAP':
                grd_global = nn.AdaptiveAvgPool1d(1)(grd_global.transpose(-1, -2)).squeeze(2)
                sat_global = nn.AdaptiveAvgPool1d(1)(sat_global.transpose(-1, -2)).squeeze(2)
            grd_global = F.normalize(grd_global, dim=1)
            sat_global = F.normalize(sat_global, dim=1)
            loss = criterion(grd_global, sat_global, args)

            #scaler.scale(loss).backward()
            loss.backward()
            #scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(itertools.chain(model_grd.parameters(), model_sat.parameters()), args.max_grad_norm,norm_type=2.0 )
            
            #scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(itertools.chain(model_grd.parameters(), model_sat.parameters()), args.max_grad_norm,norm_type=2.0 )
        
            optimizer.first_step(zero_grad=True)
            #scaler.update()
            #with torch.cuda.amp.autocast():
            
            grd_global_2 = model_grd(x_grd)
            sat_global_2 = model_sat(x_sat)
            if args.pool == 'GAP':
                grd_global_2 = nn.AdaptiveAvgPool1d(1)(grd_global_2.transpose(-1, -2)).squeeze(2)
                sat_global_2 = nn.AdaptiveAvgPool1d(1)(sat_global_2.transpose(-1, -2)).squeeze(2)
            grd_global_2 = F.normalize(grd_global_2, dim=1)
            sat_global_2 = F.normalize(sat_global_2, dim=1)
            loss_2 = criterion(grd_global_2, sat_global_2, args)

            #scaler.scale(loss_2).backward()
            loss_2.backward()
            #scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(itertools.chain(model_grd.parameters(), model_sat.parameters()), args.max_grad_norm,norm_type=2.0 )
            optimizer.second_step(zero_grad=True)
            #scaler.update()
            #optimizer.step()
            scheduler.step() 
            

            global_step += 1 

            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, len(epoch_iterator)*args.total_epoch, loss)
                )
            
            writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
        

        val_accuracy, val_accuracy_top1, val_accuracy_top5, hit_rate = valid(args, model_grd, model_sat, writer, test_loader_grd, test_loader_sat, epoch)
        if best_acc < val_accuracy_top1:
            save_model(args, model_grd, model_sat,optimizer,epoch,val_accuracy_top1)
            best_acc = val_accuracy_top1
 
        losses.reset()

    
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
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

    parser.add_argument("--pretrained_dir", type=str, default="checkpoint",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--dataset_dir", default="./CVUSA/", type=str,
                        help="The dataset path.")

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')

    # cross view setting  
    parser.add_argument("--emb_size", default=384, type=int,
                        help="embedding size")        
    
    parser.add_argument("--img_grd_size", nargs='+', default=(320, 640), type=int,
                        help="Resolution size of ground image")

    parser.add_argument("--img_sat_size", nargs='+', default=(320, 320), type=int,
                        help="Resolution size of satellite image")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
                        
    # Hyber Parameter
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for optimizer.")

    parser.add_argument("--weight_decay", default=0.03, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--start_epoch", default=0, type=int,
                        help="Start number of training epochs to perform.")

    parser.add_argument("--total_epoch", default=90, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--loss_weight", default=10, type=float,
                        help="loss_weight")

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    """
    torch.manual_seed(234)
    torch.cuda.manual_seed_all(234)    
    np.random.seed(234)
    random.seed(234)
    """

    # Model & Tokenizer Setup
    args, model_grd, model_sat = setup(args)

    # Training
    train(args, model_grd, model_sat)


if __name__ == "__main__":
    main()
