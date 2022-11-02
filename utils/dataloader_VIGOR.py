import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse
import random

norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]

class TrainDataloader(Dataset):
    def __init__(self, args):

        self.aug = True
        self.img_grd_size=args.img_grd_size
        self.img_sat_size=args.img_sat_size
        self.root = args.dataset_dir#'/data/jeff-Dataset/CV-dataset'
        self.same_area = True#False
        label_root = 'splits'
        mode = 'train_SAFA_CVM-loss-same'

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_grd_size[0], args.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((args.img_sat_size[0], args.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        if self.same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0

        # load sat list
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        
        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', train_label_fname, idx)
        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        print('Train grd loaded, data_size: {}'.format(self.train_data_size))

        

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())

        
        

    def __getitem__(self, idx):
        

        x = Image.open(self.train_list[idx]).convert('RGB')
        x = self.transform(x)
        y = Image.open(self.train_sat_list[self.train_label[idx][0]]).convert('RGB')
        y = self.transform_1(y)
        
        return x, y
        
    def __len__(self):
        return len(self.train_list)

    # avoid sampling overlap images
    def check_overlap(self, id_list, idx):
        output = True
        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                if i in sat_idx:
                    output = False
                    return output

        return output

class TestDataloader_grd(Dataset):
    def __init__(self, args):
        self.root = args.dataset_dir
        self.polar = 0#args.polar
        self.same_area =True# args.same_argsTrue#False#
        label_root = 'splits'
        mode = 'train_SAFA_CVM-loss-same'

        if self.same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_grd_size[0], args.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )
      

        self.__cur_test_id = 0
        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        self.test_sat_index_dict = {}

        self.test_sat_list = []
  
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', test_sat_list_fname, idx)

        idx = 0
        for city in self.test_city_list:
            # load test panorama list
            test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.test_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.test_label.append(label)
                    self.test_delta.append(delta)
                    if not label[0] in self.test_sat_cover_dict:
                        self.test_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', test_label_fname, idx)
        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        print('Test grd loaded, data size: {}'.format(self.test_data_size))

    def __getitem__(self, idx):

        x = Image.open(self.test_list[idx]).convert('RGB')
        x = self.transform(x)


        return x

    def __len__(self):
        return len(self.test_list)

class TestDataloader_sat(Dataset):
    def __init__(self, args):
        self.root = args.dataset_dir
        self.aug = False
        self.same_area = True#args.same_args#False#
        label_root = 'splits'
        mode = 'train_SAFA_CVM-loss-same'

        if self.same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']


        self.transform = transforms.Compose(
            [transforms.Resize((args.img_sat_size[0], args.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))


    def __getitem__(self, idx):

        y = Image.open(self.test_sat_list[idx]).convert('RGB')
        y = self.transform(y)

        return y

    def __len__(self):
        return len(self.test_sat_list)


    

        