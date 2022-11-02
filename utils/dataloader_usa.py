import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import scipy.io as sio
import torchvision
import argparse
import torch


norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
#norm_mean = [0.485, 0.456, 0.406]
#norm_std = [0.229, 0.224, 0.225]

class TrainDataloader(DataLoader):
    def __init__(self, args):
        
        self.aug = True
        self.polar = args.polar
        self.img_root = args.dataset_dir
        self.train_list = self.img_root + 'splits/train-19zl.csv'

        self.img_grd_size = args.img_grd_size
        self.img_sat_size = args.img_sat_size

        self.transform = transforms.Compose(
            [transforms.Resize((self.img_grd_size[0], self.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        
        self.transform_sat = transforms.Compose(
            [transforms.Resize((self.img_sat_size[0], self.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        
        self.__cur_id = 0 
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                if self.polar:
                    item1 = self.img_root + data[0]#.replace('bing', 'polar').replace('jpg', 'png')
                else:
                    item1 = self.img_root + data[0].replace('bingmap', 'bingmap_ori')

                item2 = self.img_root + data[1]

                self.id_list.append([item1, item2, pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)

        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

    def __getitem__(self, idx):

        
        x = Image.open(self.id_list[idx][1]).convert('RGB')
        x = self.transform(x)
        if self.polar and self.aug:
            lp = torch.randint(0,self.img_grd_size[1],(1,))
            x = x.repeat(1,1,2)
            x = x[:,:,lp:lp+self.img_grd_size[1]]
            
        
        y = Image.open(self.id_list[idx][0]).convert('RGB')
        y = self.transform_sat(y)
        if self.polar and self.aug:
            y = y.repeat(1,1,2)
            y = y[:,:,lp:lp+self.img_sat_size[1]]

        return x, y

    def __len__(self):
        return len(self.id_list)

class TestDataloader(DataLoader):
    def __init__(self, args):
        self.polar = args.polar

        self.img_root = args.dataset_dir
        self.test_list = self.img_root + 'splits/val-19zl.csv'

        self.img_grd_size = args.img_grd_size
        self.img_sat_size = args.img_sat_size

        self.transform = transforms.Compose(
            [transforms.Resize((self.img_grd_size[0], self.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )
    
        self.transform_sat = transforms.Compose(
            [transforms.Resize((self.img_sat_size[0], self.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                if self.polar:
                    item1 = self.img_root + data[0]#.replace('bing', 'polar').replace('jpg', 'png')
                else:
                    item1 = self.img_root + data[0].replace('bingmap', 'bingmap_ori')
                
                item2 = self.img_root + data[1]

                self.id_test_list.append([item1, item2, pano_id])
                
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)


    def __getitem__(self, idx):
        
        x = Image.open(self.id_test_list[idx][1]).convert('RGB')
        x = self.transform(x)  
        
        y = Image.open(self.id_test_list[idx][0]).convert('RGB')
        y = self.transform_sat(y)
            
        return x, y

    def __len__(self):
        return len(self.id_test_list)

