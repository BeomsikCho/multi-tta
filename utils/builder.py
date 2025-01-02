from typing import Optional
import transformers
from torch.utils.data import DataLoader

import inspect

import torch 

from torch import optim
import datasets
import trainers
import models

from utils.common import device_seperation

class Builder(object):
    def __init__(self, cfgs: Optional[dict] = None):
        if cfgs:
            self.cfgs = cfgs

    def build_model(self,
                    model: Optional[str] = None,
                    **model_cfgs):
        if model == None:
            model_name = self.cfgs['model']['name']
        elif not model_cfgs:
            model_cfgs = self.cfgs['model']

        if model_name == 'resnet50':
            model = models.ResNet50RobustBench()
        elif model_name == 'resnet50-gn':
            model = models.ResNet50GN()
        elif model_name == 'vit-base':
            model = models.ViTBase16()
        
        first_device, device_ids = device_seperation(self.cfgs['device'])

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            model.to(first_device)
        return model

    def build_dataloaders(self,
                          dataset: Optional[str] = None,
                          **dataset_cfgs):
        if dataset == None:
            dataset = self.cfgs['dataset']['name']
        if not dataset_cfgs:
            dataset_cfgs = self.cfgs['dataset']
        
        dataset_cls = getattr(datasets, dataset)
        for dataset in dataset_cls.build(**dataset_cfgs):
            dataloader = DataLoader(dataset = dataset,
                                    batch_size = dataset_cfgs['batch_size'],
                                    shuffle = dataset_cfgs['shuffle'],
                                    num_workers = dataset_cfgs['num_workers'])
            dataloader.name = dataset.name
            yield dataloader
        
    def build_optimizer(self,
                        optimizer: Optional[str] = None,
                        **optimizer_cfgs):
        if not optimizer:
            optimizer = self.cfgs['optimizer']['name']
        if not optimizer_cfgs:
            optimizer_cfgs = self.cfgs['optimizer']

        optimizer_cls = getattr(optim, optimizer)
        return optimizer_cls


if __name__ == "__main__":
    # Test the operation of Builder class
    builder = Builder()
    model = builder.build_model('resnet-50')
    

