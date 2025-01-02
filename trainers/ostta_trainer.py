import torch.nn as nn
from torch.optim.optimizer import Optimizer
import re
import wandb
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils.builder import Builder
from utils.metric import softmax_entropy
from utils.common import device_seperation


class OsttaTrainer(BaseTrainer):
    name = 'OsttaTrainer'

    def train_step(self, model, dataloader, optimizer):
        first_device, _ = device_seperation(self.device)
        
        model = self.configure_model(model, first_device)
        params, _ = self.collect_params(model)
        optimizer = self.adapt_optimizer(params, optimizer)

        train_loss = 0
        total_sample = 0 
        
        optimizer.zero_grad()
        for (samples, target, domain_id) in tqdm(dataloader):
            samples, target = samples.to(first_device), target.to(first_device)
            
            pred = model(samples)
            loss = criterion(pred['logits'])
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad() 

            train_loss += loss.item()
            num_sample = target.shape[0]
            total_sample += num_sample
            wandb.log({
                "step_train_loss": loss.item() / num_sample
            })

        return model

    @staticmethod
    def configure_model(model,
                        device: str,
                        adapt_layers = ['BatchNorm2d', 'GroupNorm', 'LayerNorm']):
        model.train()
        model = model.to(device)
        
        model.requires_grad_(False)
        for adapt_layer in adapt_layers:
            try:
                layer_class = getattr(nn, adapt_layer)
            except:
                raise ValueError(f"No such layer type {adapt_layer} in the torch.nn")

            for module in model.modules():
                if isinstance(module, layer_class):
                    module.requires_grad_(True)

                    if re.search('BatchNorm', adapt_layer):
                        module.track_running_stats = False
                        module.running_mean = None
                        module.running_var = None
        return model

    @staticmethod
    def collect_params(model,
                       adapt_layer = ['BatchNorm2d', 'GroupNorm', 'LayerNorm']):
        adapt_layer = tuple(map(lambda x: getattr(nn, x), adapt_layer))

        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, adapt_layer):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def adapt_optimizer(params,
                        optimizer: Optimizer,
                        **optim_cfgs
                        ) -> Optimizer:
        name_removed_optim_cfgs = {key: value for key, value in optim_cfgs.items() if key != 'name'}
        return optimizer(params, **name_removed_optim_cfgs)
    

def criterion(logits):
    marginal_coeff = 0.1
    
    entropy = softmax_entropy(logits)
    marginal_entropy = entropy.mean(0, keepdim=True)
    return (entropy - marginal_coeff * marginal_entropy).mean(0)
