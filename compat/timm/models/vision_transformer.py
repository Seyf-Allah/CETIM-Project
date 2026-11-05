import torch
from torch import nn
class DropPath(nn.Module):
    def __init__(self,drop_prob=0.):
        super().__init__();self.drop_prob=float(drop_prob)
    def forward(self,x):
        if self.drop_prob==0. or not self.training:return x
        keep=1.-self.drop_prob
        mask=(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),device=x.device,dtype=x.dtype)<keep)
        return x*mask/keep
class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__();hidden_features=hidden_features or in_features;out_features=out_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.drop1=nn.Dropout(drop)
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop2=nn.Dropout(drop)
    def forward(self,x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
