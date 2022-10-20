# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.model import twoviewmodel
from timm.models.SAIG import SAIG_Deep, SAIG_Shallow, resize_pos_embed

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, ConstantLRSchedule
from utils.data_utils import get_loader
from utils.loss import triplet_loss
import math
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

        
def save_model(args, model, optimizer, epoch, best_acc):

    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "model_checkpoint.pth")
    checkpoint = {
        'model':model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':epoch,
        'best_acc':best_acc
    }
    torch.save(checkpoint, model_checkpoint)
    
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

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
    
    state_dict = torch.load(args.pretrained_dir, map_location=torch.device('cpu') )

    # resize position embedding 
    temp = state_dict['state_dict']['pos_embed']

    state_dict['state_dict']['pos_embed'] = resize_pos_embed(temp, model_grd.pos_embed,gs_new = (args.img_grd_size[0]//16, args.img_grd_size[1]//16))
    model_grd.load_state_dict(state_dict['state_dict'],strict=False)

    state_dict['state_dict']['pos_embed'] = resize_pos_embed(temp, model_sat.pos_embed,gs_new = (args.img_sat_size[0]//16, args.img_sat_size[1]//16))
    model_sat.load_state_dict(state_dict['state_dict'],strict=False)


    model = twoviewmodel(model_grd, model_sat, args)

    if torch.cuda.device_count() >1:
        print("Using ", torch.cuda.device_count(),"GPUs!")
        model = nn.DataParallel(model)
    
    model.cuda()

    num_params = count_parameters(model)


    logger.info("Training parameters %s", args)

    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def valid(args, model, writer, test_loader, epoch):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    

    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = triplet_loss()

    sat_global_descriptor = torch.zeros([8884, args.emb_size]).cuda()
    grd_global_descriptor = torch.zeros([8884, args.emb_size]).cuda()
    val_i =0

    model.eval()
    with torch.no_grad():
        for step, (x_grd, x_sat) in enumerate(epoch_iterator):

            x_grd, x_sat=x_grd.cuda(), x_sat.cuda()

            with torch.cuda.amp.autocast():
                grd_global,sat_global = model(x_grd, x_sat)

                eval_loss = loss_fct(grd_global, sat_global, args)
            eval_losses.update(eval_loss.item())

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach()#.cpu()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach()#.cpu()
            val_i += sat_global.shape[0]

        
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    print('   compute accuracy')
    accuracy_1 = 0.0
    accuracy_5 = 0.0
    accuracy_10 = 0.0
    accuracy_89 = 0.0

    data_amount = 0.0
    dist_array = 2.0 - 2.0 * torch.matmul(sat_global_descriptor, grd_global_descriptor.T)
    print('start')
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = torch.sum(dist_array[:, i] < gt_dist)
        if prediction < 1:
            accuracy_1 += 1.0
        if prediction < 5:
            accuracy_5 += 1.0
        if prediction < 10:
            accuracy_10 += 1.0
        if prediction < 89:
            accuracy_89 += 1.0
        data_amount += 1.0
    accuracy_1 /= data_amount
    accuracy_5 /= data_amount
    accuracy_10 /= data_amount
    accuracy_89 /= data_amount

    print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f' % (accuracy_1*100.0, accuracy_5*100.0, accuracy_10*100.0, accuracy_89*100.0))

    # save eval result
    file = './Result/'+ args.dataset + '/' + str(args.model_type) + '_accuracy.txt'
    if not os.path.exists('./Result/'+ args.dataset):
        os.makedirs('./Result/'+ args.dataset)
    with open(file, 'a') as file:
        file.write(str(epoch) + ' ' + ' : ' + str(accuracy_1*100.0) + '  '+ str(accuracy_5*100.0)+ '  '+ str(accuracy_10*100.0) + '  '+ str(accuracy_89*100.0)  + '\n')

    # print the valid information
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy_1)

    writer.add_scalar("test/accuracy", scalar_value=accuracy_1, global_step=epoch)

    return accuracy_1

def train(args, model):
    """ Train the model """
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))


    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)
                        
    
    
    if args.resume:
        print(args.resume)
        state_dict = torch.load(os.path.join(args.resume,'model_checkpoint.pth'), map_location='cpu' )
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        global_step = start_epoch * len(train_loader)
        best_acc = state_dict['best_acc']

        print('model start from ' + str(start_epoch) + ' epoch')
        print('best acc: ' + str(best_acc))
    else:
        # load pretrained model
        start_epoch = args.start_epoch
        global_step, best_acc = 0, 0
    
    t_total = args.total_epoch * len(train_loader)
    print(t_total)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
   
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # loss function
    criterion = triplet_loss()
    losses = AverageMeter()

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.total_epoch):
        model.train()

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        for step, (x_grd, x_sat) in enumerate(epoch_iterator):
            x_grd, x_sat=x_grd.cuda(), x_sat.cuda()
            
            #torchvision.utils.save_image(x_grd,'./1.jpg',normalize=True)
            #torchvision.utils.save_image(x_sat,'./2.jpg',normalize=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                grd_global,sat_global = model(x_grd,x_sat)

                loss = criterion(grd_global, sat_global, args)

            # backward
            scaler.scale(loss).backward()
            losses.update(loss.item())
            
            # gradient clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # optimizer.step
            scaler.step(optimizer)
            scaler.update()

            # scheduler
            scheduler.step() 
            
            global_step += 1 

            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, len(epoch_iterator)*args.total_epoch, losses.val)
                )

            # record loss&lr
            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
        
        # validate
        accuracy = valid(args, model, writer, test_loader, epoch)
        if best_acc < accuracy:
            save_model(args, model,optimizer,epoch,accuracy)
            best_acc = accuracy
 
        losses.reset()

    
    writer.close()

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

def main():
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

    parser.add_argument("--pretrained_dir", type=str, default="checkpoint",
                        help="Where to search for pretrained models.")

    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--dataset_dir", default="./CVUSA/", type=str,
                        help="The dataset path.")

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')

    # cross view setting                
    parser.add_argument("--polar", type=int,choices=[1,0],
                        default=1,
                        help="polar transform or not")

    parser.add_argument("--emb_size", default=384, type=int,
                        help="embedding size")

    parser.add_argument("--img_grd_size", nargs='+', default=(128, 512), type=int,
                        help="Resolution size of ground image")

    parser.add_argument("--img_sat_size", nargs='+', default=(256, 256), type=int,
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

    parser.add_argument("--total_epoch", default=200, type=int,
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

    assert args.polar==0 and args.img_sat_size[0] == args.img_sat_size[1] or args.polar==1 and args.img_sat_size[0] != args.img_sat_size[1], \
            f"Input sat image size ({args.img_sat_size[0]}*{args.img_sat_size[1]}) doesn't match the polar transformation."

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # Model & Tokenizer Setup
    args, model = setup(args)
  
    # Training
    train(args, model)


if __name__ == "__main__":
    main()
