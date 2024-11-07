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
        self.builder = Builder(cfgs)
        self.cfgs = cfgs['trainer']
        self.device = self.cfgs['device']
        
    def train(self):
        for dataloader in self.builder.build_dataloaders():
            model = self.builder.build_model()
            optimizer = self.builder.build_optimizer()
            
            model = self.train_step(model, dataloader, optimizer)
            result = self.validate_step(model, dataloader)
            wandb.log(result, dataloader.name)

    def validate(self):
        for dataloader in self.builder.build_dataloaders():
            model = self.builder.build_model()
    
            result = self.validate_step(model, dataloader)
            wandb.log(result, dataloader.name)

    def train_step(self, model, dataloader):
        pass

    def validate_step(self, model, dataloader):
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.cfgs[key] = value