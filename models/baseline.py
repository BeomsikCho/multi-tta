import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

import timm
from robustbench.utils import load_model

from abc import *

class MetaModel(metaclass=ABCMeta):
    @abstractmethod
    def forward() -> dict:
        pass

    def validate() -> dict:
        pass

class ResNet50RobustBench(nn.Module):
    def __init__(self):
        super().__init__()
        total_model = load_model('Standard_R50', dataset='imagenet', threat_model='corruptions')
        processor, model = list(total_model.children())
        
        self.processor = processor
        self.encoder = nn.Sequential(*list(model.children()))[:-2]
        self.fc = nn.Sequential(*list(model.children()))[-1]
    
    def forward(self, samples):
        samples = self.processor(samples)

        pred = dict()
        pred['last_hidden_state'] = self.encoder(samples)
        pred['logits'] = self.fc(adaptive_avg_pool2d(pred['last_hidden_state'], (1,1)).squeeze())
        return pred

class ResNet50GN(nn.Module):
    def __init__(self):
        super().__init__()
        total_model = timm.create_model('resnet50_gn', pretrained=True)
        self.encoder = nn.Sequential(*list(total_model.children()))[:-2]
        self.fc = nn.Sequential(*list(total_model.children()))[-1]

    def forward(self, samples):
        pred = dict()
        pred['last_hidden_state'] = self.encoder(samples)
        pred['logits'] = self.fc(adaptive_avg_pool2d(pred['last_hidden_state'], (1,1)).squeeze())
        return pred

class ViTBase16(nn.Module):
    def __init__(self):
        super().__init__()
        total_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder = nn.Sequential(*list(total_model.children()))[:-2]
        self.fc = nn.Sequential(*list(total_model.children()))[-1]
        
    def forward(self, samples):
        pred = dict()
        pred['last_hidden_state'] = self.encoder(samples)
        pred['logits'] = self.fc(adaptive_avg_pool2d(pred['last_hidden_state'], (1,1)).squeeze())
        return pred


if __name__ == "__main__":
    pass