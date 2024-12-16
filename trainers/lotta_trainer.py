import torch.nn as nn
from torch.optim.optimizer import Optimizer
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F

from typing import List


from .tent_trainer import TentTrainer
from utils.builder import Builder
from utils.metric import softmax_entropy, empirical_fisher_information
from utils.common import device_seperation

class LottaTrainer(TentTrainer):
    name = 'LottaTrainer'

    def train_step(self, model, dataloader, optimizer):
        first_device, _ = device_seperation(self.device)
        
        model = self.configure_model(model, first_device)
        params, _ = self.collect_params(model)
        optimizer = self.adapt_optimizer(params, optimizer)

        # Lotta의 Loss 설정 부분
        loss_fn = LottaLoss(e_margin= self.cfgs['e_margin'],
                          fisher_alpha= self.cfgs['fisher_alpha'],
                          d_margin= self.cfgs["d_margin"])
        train_loss = 0
        total_sample = 0 
        
        optimizer.zero_grad()
        for (samples, target, domain_id) in tqdm(dataloader):
            samples, target = samples.to(first_device), target.to(first_device)
            
            pred = model(samples)
            loss = loss_fn.total_loss(model, pred['logits']).mean(0)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad() 

            train_loss += loss.item()
            num_sample = target.shape[0]
            total_sample += num_sample
            wandb.log({
                "step_train_loss": loss.item() / num_sample,
                "step_gradient_norm": empirical_fisher_information(model)
            })

        return model
    
    @staticmethod
    def configure_model(model: nn.Module, device: str, adapt_layers) -> nn.Module:
        model = LoRA(model, adapt_layers)
        model.train()
        model = model.to(device)
        model.requires_grad_(False)
        model.set_trainable_param()
        return model

    @staticmethod
    def collect_params(model: nn.Module):
        params = []
        names = []
        for nm, m in model.lora_layers.named_modules():
            for np, p in m.named_parameters():
                params.append(p)
                names.append(f"{nm}.{np}")
        return params, names



class LoRA(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 adapt_layers: List[str]
                 ) -> None:
        super(LoRA, self).__init__()
        self.connection_type = nn.Conv2d

        self.split_layers(model)
        self.lora_layers = nn.ModuleDict()

        for layer_name in adapt_layers:
            original_layer = getattr(self, layer_name)
            first_conv = self.get_first_conv_layer(original_layer)
            last_conv = self.find_last_conv_layer(original_layer)
            in_channels = first_conv.in_channels
            out_channels = last_conv.out_channels
            
            if layer_name == "layer1":
                stride = 1
            else:
                stride = 2
            self.lora_layers[layer_name] = self.connection_type(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.initialize_lora_parameters()

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x) + 0.05 * self.lora_layers['layer1'](x)
        x = self.layer2(x) + 0.05 * self.lora_layers['layer2'](x)
        x = self.layer3(x) + 0.05 * self.lora_layers['layer3'](x)
        x = self.layer4(x) + 0.05 * self.lora_layers['layer4'](x)
        x = self.post_layer(x)
        return x

    def split_layers(self, model: nn.Module) -> None:
        self.pre_layer = nn.Sequential()
        self.post_layer = nn.Sequential()
    
        # Flags to track position in the model
        in_pre_layer = True

        for name, module in model.named_children():
            if name == 'layer1':
                in_pre_layer = False
                self.layer1 = module
            elif name == 'layer2':
                self.layer2 = module
            elif name == 'layer3':
                self.layer3 = module
            elif name == 'layer4':
                self.layer4 = module
            else:
                if in_pre_layer:
                    self.pre_layer.add_module(name, module)
                else:
                    self.post_layer.add_module(name, module)

    def get_first_conv_layer(self, layer: nn.Module) -> nn.Conv2d:
        # Recursively find the first Conv2d layer
        for name, child in layer.named_children():
            if isinstance(child, nn.Conv2d):
                return child
            result = self.get_first_conv_layer(child)
            if result is not None:
                return result
        return None
    
    def find_last_conv_layer(self, layer: nn.Module) -> nn.Conv2d:
        # Recursively find the last Conv2d layer
        
        last_conv_layer = None
        stack = [layer]
    
        while stack:
            current_module = stack.pop()
        
            if isinstance(current_module, nn.Conv2d):
                return current_module
        
            for child in current_module.children():
                stack.append(child)
        
        return last_conv_layer

    def set_trainable_param(self) -> nn.Module:
        # Fix the parameters of the original model
        for param in self.parameters():
            param.requires_grad = False
        
        # Make the parameters of the adaptive layers trainable
        for param in self.lora_layers.parameters():
            param.requires_grad = True

        return self

    def initialize_lora_parameters(self):
        for name, layer in self.lora_layers.items():
            for param in layer.parameters():
                nn.init.constant_(param, 0)




class LottaLoss():
    def __init__(self, e_margin: float, d_margin: float, fisher_alpha: float):
        self.e_margin = e_margin 
        self.d_margin = d_margin 
        self.fisher_alpha = fisher_alpha
        self.current_model_probs = None

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    

    def update_model_probs(self, new_probs: torch.Tensor) -> None:
        if self.current_model_probs is None:
            if new_probs.size(0) != 0:
                with torch.no_grad():
                    self.current_model_probs = new_probs.mean(0)
        elif new_probs.size(0) != 0:
            with torch.no_grad():
                self.current_model_probs = 0.9 * self.current_model_probs + (1 - 0.9) * new_probs.mean(0)


    def filter_by_entropy_margin(self, entropy: torch.Tensor) -> torch.Tensor:
        filter_ids1 = torch.where(entropy < self.e_margin)
        entropy = entropy[filter_ids1]
    
        coeff = 1 / (torch.exp(entropy.clone().detach() - self.e_margin))
        entropy = entropy.mul(coeff)
        return entropy, filter_ids1
    

    def filter_by_output(self, entropy: torch.Tensor, outputs: torch.Tensor, filter_ids1) -> torch.Tensor:
        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropy = entropy[filter_ids_2]
            self.update_model_probs(outputs[filter_ids1][filter_ids_2].softmax(1))
        else:
            self.update_model_probs(outputs[filter_ids1].softmax(1))
            filter_ids_2 = None
        
        return entropy, filter_ids_2


    def fisher_loss(self, model: nn.Module):
        return empirical_fisher_information(model)
    

    def total_loss(self, model: nn.Module, outputs: torch.Tensor):
        entropy = self.softmax_entropy(outputs)
        entropy, filter_ids1 = self.filter_by_entropy_margin(entropy)
        entropy, _ = self.filter_by_output(entropy, outputs, filter_ids1)
        
        return entropy.mean(0) + self.fisher_alpha * self.fisher_loss(model)