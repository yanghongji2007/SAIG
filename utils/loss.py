import torch

import torch.nn as nn
import math
from torch.nn import functional as F
def weighted_soft_margin_loss(diff, beta=10.0, reduction=torch.mean):

    out = torch.log(1 + torch.exp(diff * beta))
    if reduction:
        out = reduction(out)
    return out


def get_semi_hard_neg(logits, pos_dist):
    
    N=logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_=torch.lt(logits, pos_dist.unsqueeze(1))
    mask[targets, targets] = False
    mask_[targets, targets] = False

    mininum=torch.mul(logits, mask_)
    hard_neg_dist1, _ = torch.max(mininum, 1)
    hard_neg_dist2, _ = torch.min(logits[mask].reshape(N,-1), 1)
    hard_neg_dist = torch.max(hard_neg_dist1, hard_neg_dist2)

    return hard_neg_dist

def get_topk_hard_neg(logits, pos_dist, k):
    
    N=logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[targets, targets] = False
    hard_neg_dist, _ = torch.topk(logits[mask].reshape(N,-1), largest=False, k=k, dim=1)
    return hard_neg_dist


class SemiSoftTriHard(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, grd_global, sat_global, args):

        dist_array = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diag(dist_array)

        logits = dist_array
        #hard_neg_dist_g2s = get_semi_hard_neg(logits, pos_dist)
        #hard_neg_dist_s2g = get_semi_hard_neg(logits.t(), pos_dist)
        hard_neg_dist_g2s = get_topk_hard_neg(logits, pos_dist, int(args.train_batch_size**0.5))
        hard_neg_dist_s2g = get_topk_hard_neg(logits.t(), pos_dist, int(args.train_batch_size**0.5))
        return (weighted_soft_margin_loss(pos_dist-hard_neg_dist_g2s.t(), args.loss_weight) + weighted_soft_margin_loss(pos_dist-hard_neg_dist_s2g.t(), args.loss_weight))/2.0

             
class triplet_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, grd_global, sat_global, args):
        dist_array = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.T)
        
        pos_dist = torch.diag(dist_array)
        pair_n = args.train_batch_size * (args.train_batch_size - 1.0)
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_g2s * args.loss_weight)))/pair_n
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        loss_s2g = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_s2g * args.loss_weight)))/pair_n
        loss = (loss_g2s + loss_s2g) / 2.0
        
        return loss


class InfoNCE(nn.Module):
    def __init__(self, t=0.02):
        super().__init__()
        self.t=t
    def forward(self, grd_global, sat_global, args):
        N = grd_global.shape[0]
        logits = sat_global @ grd_global.t()
        logits = torch.cat((logits, logits.t()), dim=1)
        targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[targets, targets+N] = False
        logits = logits[mask].reshape(N, -1)
        return F.cross_entropy(logits/self.t, targets)



if __name__=="__main__":
    loss=InfoNCE()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    
    parser.add_argument("--loss_weight", default=10, type=float,
                        help="loss_weight")
    args = parser.parse_args()

    grd_global = F.normalize(torch.rand(32, 384), dim=1)
    sat_global1 = F.normalize(torch.rand(32, 384), dim=1)
    sat_global2 = F.normalize(torch.rand(32, 384), dim=1)
    print(loss(grd_global, sat_global1, args))
    """
    a=torch.tensor([[0, 1,2,3,4,5,6,7] for _ in range(14)])
    b=torch.tensor([[1, 3,4,5,6,7,2,1]])
    mask=torch.lt(a, b)
    print(mask)
    """
    