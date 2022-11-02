import torch
import torch.nn as nn
from torch.nn import functional as F

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

class ReduceDimBranch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.pooling = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.reduceLinear = nn.Conv2d(384, 48, 1 , 1, 0)

    def forward(self, x):
        
        H, W = x.shape[2]//16, x.shape[3]//16

        x = self.backbone(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # [B, 384, 8, 32]
        x = self.pooling(x) # [B, 384, 4, 16]
        x = self.reduceLinear(x) # [B, 48, 4, 16]

        #x = torch.flatten(x, start_dim=-3, end_dim=-1) # [B, 48*4*16]
        x = x.permute(0,2,3,1).flatten(1,2).flatten(1,2)
        return x


class MixerHead_Branch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        
        self.linear1 = nn.Linear(256, 1024)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 8)
        """
        self.linear = nn.Linear(256, 8)"""
        
        
    def forward(self, x):
        
        #H, W = x.shape[2]//16, x.shape[3]//16

        x = self.backbone(x)
        """
        #x = x.flatten(-2,-1)
        x = x.transpose(-1,-2)
        x = self.linear(x)
        """
        x = x.transpose(-1,-2)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.transpose(-1,-2) # [B, 8, 384]
        
    
        x = x.flatten(-2,-1)
        return x
class SAFABranch(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.backbone = model
        self.pooling = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.spatial_aware = SpatialAware(64, d=8)
        
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


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class GeMBranch(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.backbone = model
        self.pooling = GeM(p=3)
        #self.pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):

        H, W = x.shape[2]//16, x.shape[3]//16
        x = self.backbone(x) # [B, 256, 384]

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # [B, 384, 8, 32]
        x = self.pooling(x).squeeze() # [B, 384, 1]
        
        return x

class Branch(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.backbone = model
        self.pooling = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        
        x = self.backbone(x) # [B, 256, 384]
        x = self.pooling(x.transpose(1, 2)).squeeze(2)

        return x

class twoviewmodel(nn.Module):
    def __init__(self,model_grd,model_sat,args):
        super().__init__()
        # self.model_grd = Branch(model_grd)
        # self.model_sat = Branch(model_sat)
        if args.pool == 'GAP':
            self.model_grd = Branch(model_grd)
            self.model_sat = Branch(model_sat)
        elif args.pool == 'SMD':
            self.model_grd = MixerHead_Branch(model_grd)
            self.model_sat = MixerHead_Branch(model_sat)
        print(self.model_grd)

    def forward(self,x_grd,x_sat):
        x_grd = self.model_grd(x_grd)
        x_sat = self.model_sat(x_sat)
        return F.normalize(x_grd, dim=1), F.normalize(x_sat, dim=1)
        

