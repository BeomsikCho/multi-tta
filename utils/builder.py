from typing import Optional
import transformers
from torch.utils.data import DataLoader

import inspect

from torch import optim
import datasets
import trainers

class Builder(object):
    def __init__(self, cfgs: Optional[dict] = None):
        if cfgs:
            self.cfgs = cfgs

    def build_model(self,
                    model: Optional[str] = None,
                    pretrained: bool = True,
                    **model_cfgs):
        if model == None:
            model = self.cfgs['model']['name']
        elif not model_cfgs:
            model_cfgs = self.cfgs['model']

        if model == 'resnet-50':
            if pretrained:
                model= transformers.ResNetModel.from_pretrained('microsoft/resnet-50') # input should 
                processor = transformers.AutoImageProcessor.from_pretrained('microsoft/resnet-50')
            else:
                model= transformers.ResNetModel(config='microsoft/resnet-50')
                processor = transformers.AutoImageProcessor(config='microsoft/resnet-50')

        elif model == 'vit-base':
            if pretrained:
                model= transformers.ResNetModel.from_pretrained('google/vit-base-patch16-224-in21k') # input should 
                processor = transformers.AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            else:
                model= transformers.ResNetModel(config='google/vit-base-patch16-224-in21k')
                processor = transformers.AutoImageProcessor(config='google/vit-base-patch16-224-in21k')

        return model, processor

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
    model, processor = builder.build_model('resnet-50')
    breakpoint()
    

