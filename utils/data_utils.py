import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def get_loader(args):

    if args.dataset == 'CVUSA':
        from utils.dataloader_usa import TrainDataloader,TestDataloader
    elif args.dataset == 'CVACT':
        from utils.dataloader_act import TrainDataloader,TestDataloader
    elif args.dataset == 'VIGOR':
        from utils.dataloader_VIGOR import TrainDataloader,TestDataloader_grd, TestDataloader_sat

    trainset = TrainDataloader(args)
    train_loader = DataLoader(trainset,
                            batch_size=args.train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=4)
                            
    if args.dataset == 'VIGOR':
        testset_grd = TestDataloader_grd(args)
        test_loader_grd = DataLoader(testset_grd,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=4,
                            drop_last=False)

        testset_sat = TestDataloader_sat(args)
        test_loader_sat = DataLoader(testset_sat,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=4,
                            drop_last=False)
        
        return train_loader, test_loader_grd, test_loader_sat

    testset = TestDataloader(args)
    test_loader = DataLoader(testset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=4,
                            drop_last=False)


    return train_loader, test_loader

