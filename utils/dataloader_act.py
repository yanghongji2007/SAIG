import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import scipy.io as sio
import torchvision
import argparse
import torch
__all__ = ['TrainDataloader','TestDataloader']


norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
#norm_mean = [0.485, 0.456, 0.406]
#norm_std = [0.229, 0.224, 0.225]


class TrainDataloader(DataLoader):
    def __init__(self, args):
        
        self.aug = False
        self.polar = args.polar

        self.img_grd_size = args.img_grd_size
        self.img_sat_size = args.img_sat_size

        self.img_root = args.dataset_dir 
        self.transform = transforms.Compose(
            [transforms.Resize((self.img_grd_size[0], self.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )
        
        self.transform_sat = transforms.Compose(
            [transforms.Resize((self.img_sat_size[0], self.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        self.allDataList = './ACT_data.mat'

        __cur_allid = 0  # for training
        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)


        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            """
            if self.polar:
                # polar transform and crop the ground view
                grd_id_align = self.img_root + 'polar_streetview1/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'polar_map1/' + anuData['panoIds'][i] + '_satView_polish.png'
            else:
                grd_id_align = self.img_root + 'polar_streetview1/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
            """
            if self.polar:
                grd_id_align = self.img_root + 'polar_streetview1/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'polar_map1/' + anuData['panoIds'][i] + '_satView_polish.png'
                # grd_id_align = self.img_root + 'streetview_polar/' + anuData['panoIds'][i] + '_grdView.png'
                # sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'
            else:
                grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
                # grd_id_align = self.img_root + 'streetview_polar/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'
            
            id_alllist.append([ grd_id_align, sat_id_ori])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)



        training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        trainNum = len(training_inds)
        print('trainSet:' ,trainNum)
        self.trainList = []
        self.trainIdList = []

        
        for k in range(trainNum):
           
            self.trainList.append(id_alllist[training_inds[k][0]])
            self.trainIdList.append(k)  


    def __getitem__(self, idx):

        
        x = Image.open(self.trainList[idx][0]).convert('RGB')
        x = self.transform(x) 

        if self.polar and self.aug:
            lp = torch.randint(0,self.img_grd_size[1],(1,))
            x = x.repeat(1,1,2)
            x = x[:,:,lp:lp+self.img_grd_size[1]]


        y = Image.open(self.trainList[idx][1]).convert('RGB')
        y = self.transform_sat(y)
        if self.polar and self.aug:
            y = y.repeat(1,1,2)
            y = y[:,:,lp:lp+self.img_sat_size[1]]


        return x, y

    def __len__(self):
        return len(self.trainList)


class TestDataloader(DataLoader):
    def __init__(self, args):

        self.polar = args.polar

        self.img_root = args.dataset_dir

        self.img_grd_size = args.img_grd_size
        self.img_sat_size = args.img_sat_size

        self.transform = transforms.Compose(
            [transforms.Resize((self.img_grd_size[0], self.img_grd_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std) ] )

        self.transform_sat = transforms.Compose(
            [transforms.Resize((self.img_sat_size[0], self.img_sat_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = norm_mean, std = norm_std)] )

        self.allDataList = './ACT_data.mat'

        __cur_allid = 0  # for training
        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)


        idx = 0
        for i in range(0,len(anuData['panoIds'])):
    
            if self.polar:
                grd_id_align = self.img_root + 'polar_streetview1/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'polar_map1/' + anuData['panoIds'][i] + '_satView_polish.png'
                # grd_id_align = self.img_root + 'streetview_polar/' + anuData['panoIds'][i] + '_grdView.png'
                # sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'
            else:
                # grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
                grd_id_align = self.img_root + 'polar_streetview1/' + anuData['panoIds'][i] + '_grdView.png'
                sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
            
            id_alllist.append([ grd_id_align, sat_id_ori])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)


        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)
        print('valSet:' ,self.valNum)
        self.valList = []

        for k in range(self.valNum):
            self.valList.append(id_alllist[self.val_inds[k][0]])

        self.__cur_test_id = 0      

    def __getitem__(self, idx):
        x = Image.open(self.valList[idx][0]).convert('RGB')
        x = self.transform(x)
        
        y = Image.open(self.valList[idx][1]).convert('RGB')
        y = self.transform_sat(y)

        return x, y

    def __len__(self):
        return len(self.valList)


