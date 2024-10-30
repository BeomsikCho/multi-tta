from typing import Optional
import transformers

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

    def build_dataloder(self, dataset: Optional[str]):
        # DataLoader의 output은 [data, target, domain_id]순으로 작성됨 
        if dataset == None:
            dataest = self.cfgs['dataset']

        if dataset == 'imagenet-c':
            dataset = datasets.ImageNetC()

    def build_optimizer(self, optimizer: Optional[str]):
        pass

    def build_trainer(self, trainer: Optional[str] = None):
        if trainer == None:
            trainer = self.cfgs['trainer']

        assert trainer in ['base, tent, eata, sar, emt']

        if trainer == 'base':
            return trainers.BaseTrainer(self.cfgs)
        # elif trainer == 'tent':
        #     trainer = trainers.TentTrainer(self.cfgs)
        # elif trainer == 'eata':
        #     trainer = trainers.EataTrainer(self.cfgs)
        # elif trainer == 'tent':
        #     trainer = trainers.SarTrainer(self.cfgs)
        # elif trainer == 'tent': 
        #     trainer = trainers.EmtTrainer(self.cfgs) # ours



if __name__ == "__main__":
    # Test the operation of Builder class
    builder = Builder()
    model, processor = builder.build_model('resnet-50')
    breakpoint()
    

