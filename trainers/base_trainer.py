from abc import *
from tqdm import tqdm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import torch

from utils import Builder
from utils.common import device_seperation

class MetaTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train() -> dict:
        pass

    @abstractmethod
    def validate() -> dict:
        pass


class BaseTrainer(MetaTrainer):
    name = 'BaseTrainer'

    def __init__(self, cfgs):
        self.builder = Builder(cfgs)
        self.device = cfgs['device']
        self.cfgs = cfgs['trainer']
        
        # Initialize wandb once
        wandb.login()
        wandb.init(
            project='multiTTA',
            name=self.name,
            config=cfgs
        )

    def train(self):
        accuracy_data = []
        for dataloader in self.builder.build_dataloaders():
            model = self.builder.build_model()
            optimizer = self.builder.build_optimizer()
        
            model_name = getattr(model, 'name', 'default_model')
            dataloader_name = getattr(dataloader, 'name', 'default_dataloader')

            model = self.train_step(model, dataloader, optimizer)        
            results = self.validate_step(model, dataloader)
        
            # Append the results to the data list
            accuracy_data.append({
                'model_name': model_name,
                'dataloader_name': dataloader_name,
                'top1_acc': results['top1_acc']
            })

            # Convert the data to a DataFrame
            df = pd.DataFrame(accuracy_data)

            fig = px.bar(
                df,
                x='dataloader_name',
                y='top1_acc',
                color='dataloader_name',
                title='Accuracy per Domain',
                labels={'dataloader_name': 'Dataloader Name', 'top1_acc': 'Top-1 Accuracy'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                legend_title_text='Dataloader Name',
                xaxis_title=None,
                yaxis_title='Top-1 Accuracy',
                title_x=0.5,
                autosize=True  # 반응형 설정
            )
            # WandB에 Plotly 그래프 로그
            wandb.log({"domain_accuracy": fig})

            del dataloader
            del model

    def validate(self):
        for dataloader in self.builder.build_dataloaders():
            model = self.builder.build_model()
    
            result = self.validate_step(model, dataloader)
            wandb.log(result, dataloader.name)

    def train_step(self, model, dataloader, optimizer):
        pass
    
    def validate_step(self, model, dataloader):
        first_device, _ = device_seperation(self.device)
        model.eval()
        is_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for (samples, target, domain_id) in tqdm(dataloader):
                samples, target = samples.to(first_device), target.to(first_device)
                pred = model(samples)

                is_correct += (pred['logits'].argmax(dim=1) == target).sum().item()
                total_samples += target.size(0)

        results = dict()
        results['top1_acc'] = is_correct / total_samples
        return results
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.cfgs[key] = value