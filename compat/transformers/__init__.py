import json
from torch import nn
class PretrainedConfig:
    def __init__(self,**kwargs):
        for k,v in kwargs.items():setattr(self,k,v)
    @classmethod
    def from_json_file(cls,file):
        with open(file,encoding='utf-8') as f:return cls(**json.load(f))
class PreTrainedModel(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
