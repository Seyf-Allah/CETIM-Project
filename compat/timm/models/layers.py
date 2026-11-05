from torch.nn.init import trunc_normal_
def to_2tuple(x):
    return x if isinstance(x,(tuple,list)) else (x,x)
