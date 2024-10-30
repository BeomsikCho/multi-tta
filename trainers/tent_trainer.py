import torch.nn as nn
from torch.optim.optimizer import Optimizer
import re
import wandb

from .base_trainer import BaseTrainer
from utils import Builder
from utils import softmax_entropy

class TentTrainer(BaseTrainer):
    def train_step(self, model, device, processor, dataloader, optimizer):
        model = self.configure_model(model, device)
        params, param_names = self.collect_params(model)
        optimizer = self.adapt_optimizer(params, optimizer)

        train_loss = 0
        correct = 0
        total_sample = 0
        for iteration, (samples, target, domain_id) in enumerate(dataloader):
            samples = processor(samples)
            pred = model(samples['pixel_values'])
            loss = softmax_entropy(pred)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            correct += (pred.argmax(dim=1) == target).sum().item()
            total_sample += target.size(0)

            wandb.log({
                "step_train_loss": loss.item(), # 현재 loss
                "cumulative_train_loss": train_loss / (iteration + 1) # loss의 평균
            })
        accuracy = correct / total_sample
        wandb.log({
            "train_loss": train_loss / len(dataloader),  # 평균 훈련 손실
            "accuracy": accuracy  # 평균 정확도
        })
        return model


    @staticmethod
    def configure_model(model, device: str, adapt_layers):
        model.train()
        model = model.to(device)

        if adapt_layers != None: return model

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
    def collect_params(model):
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
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
    

    def validate_step(self, model, dataloader) -> dict:
        pass

