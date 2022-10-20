import os
import torch
import torch.nn as nn
import numpy as np



from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import argparse
#
state_dict = torch.load('/raid/zhuyingying/swin_transformer/pytorch-image-models-master/output/20210914-214648-pvt_small-224/model_best.pth.tar',map_location=torch.device('cpu'))


for k in state_dict['state_dict']:
    print(k, state_dict['state_dict'][k].shape)
