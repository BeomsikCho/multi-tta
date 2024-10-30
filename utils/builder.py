from typing import Optional
import transformers
from torch.utils.data import DataLoader

import datasets
import trainers

class Builder(object):
    def __init__(self, cfgs: Optional[dict] = None):
        if cfgs:
            self.cfgs = cfgs

    def build_model(self, model: Optional[str], pretrained: bool = True):
        if model == None:
            model = self.cfgs['model']

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
                          dataset: Optional[str],
                          **dataset_cfgs):
        if dataset == None:
            dataset = self.cfgs['dataset']['name']
        if dataset_cfgs == None:
            dataset_cfgs = dataset['dataset']
        
        dataset_class = getattr(datasets, dataset)
        for dataset in dataset_class.build(**dataset_cfgs):
            dataloader = DataLoader(dataset = dataset,
                                    batch_size = dataset_cfgs['batch_size'],
                                    shuffle = dataset_cfgs['shuffle'],
                                    num_workers = dataset_cfgs['num_workers'])
        dataloader.name = dataset.name
        yield dataloader
        
    def build_optimizer(self, optimizer: Optional[str]):
        pass


if __name__ == "__main__":
    # Test the operation of Builder class
    builder = Builder()
    model, processor = builder.build_model('resnet-50')
    breakpoint()
    

