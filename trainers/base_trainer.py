from abc import *
# from utils import Builder

class MetaTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def validate():
        pass


class BaseTrainer(MetaTrainer):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        # self.builder = Builder(cfgs)
        
        # self.model, self.processor = self.builder.build_model()
        # self.dataloader = self.builder.build_dataloder()
        # self.optimizer = self.builder.build_optimizer()

    def train(self):
        for epoch in self.cfgs['epochs']:
            self.model.train()
            for sample in self.dataloader:
                # sample은 ['pixel_values', 'target', 'domain_id']을 반드시 포함하고 있어야 함.
                sample['domain_id'] 

    def validate(self):
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.cfgs[key] = value