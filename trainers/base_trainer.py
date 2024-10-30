from abc import *
import wandb

from utils import Builder

class MetaTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train() -> dict:
        pass

    @abstractmethod
    def validate() -> dict:
        pass


class BaseTrainer(MetaTrainer):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.builder = Builder(cfgs)
        
    def train(self):
        
        for dataloader in self.builder.build_dataloaders():
            model, processor = self.builder.build_model()
            optimizer = self.builder.build_optimizer()
            model = self.train_step(model, processor, dataloader, optimizer)
            result = self.validate_step(model, dataloader)
            wandb.log(result, dataloader.name)

    def validate(self):
        pass
    
    def train_step(self, model, dataloader):
        pass

    def validate_step(self, model, dataloader):
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.cfgs[key] = value