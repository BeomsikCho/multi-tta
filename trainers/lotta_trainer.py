import torch.nn as nn
from torch.optim.optimizer import Optimizer
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F

from typing import List

# Added for visaulization
import matplotlib.pyplot as plt
import numpy as np


from .tent_trainer import TentTrainer
from utils.builder import Builder
from utils.metric import softmax_entropy, empirical_fisher_information
from utils.common import device_seperation

class LottaTrainer(TentTrainer):
    name = 'LottaTrainer'

    def train_step(self, model, dataloader, optimizer):
        first_device, _ = device_seperation(self.device)
        
        model = self.configure_model(model, first_device, self.cfgs['lora_alpha'], self.cfgs['connection_type'], self.cfgs['adapt_layers'])
        params, _ = self.collect_params(model)

        print("Enter Optimizer")
        optimizer = self.adapt_optimizer(params, optimizer)
        print("Out Optimizer")

        # Lotta의 Loss 설정 부분
        loss_fn = LottaLoss(e_margin= self.cfgs['e_margin'],
                          fisher_alpha= self.cfgs['fisher_alpha'],
                          d_margin= self.cfgs["d_margin"])
        train_loss = 0
        total_sample = 0 
        
        optimizer.zero_grad()
        for idx, (samples, target, domain_id) in tqdm(enumerate(dataloader)):
            samples, target = samples.to(first_device), target.to(first_device)
            
            pred = model(samples)
            loss = loss_fn.total_loss(model, pred['logits']).mean(0)
            loss.backward()
            
            train_loss += loss.item()
            num_sample = target.shape[0]
            total_sample += num_sample
            
            # Record the train procedure
            grad_norm = empirical_fisher_information(model)
            # if idx != 0 and grad_norm > 1.4 * prev_norm:
            #     self.visualize_samples(idx, samples, target, domain_id, pred)

            wandb.log({
                "step_train_loss": loss.item() / num_sample,
                "step_gradient_norm": empirical_fisher_information(model)
            })

            prev_norm = grad_norm
            optimizer.step()
            optimizer.zero_grad() 

        return model


    def visualize_samples(self, idx, samples, target, domain_id, pred):
        with open("./data/imagenet_classes.txt", "r") as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
    
        worst_sample = samples.cpu().detach().numpy()               # (N, C, H, W)
        target_np = target.cpu().detach().numpy()                   # (N,)
        pred_np = pred['logits'].argmax(1).cpu().detach().numpy()   # (N,)
    
        N = worst_sample.shape[0]
    
        import math
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
    
        fig_width = 12
        fig_height_per_row = 4
        fig_height = fig_height_per_row * rows
    
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height), dpi=100)
        axes = axes.flatten() 
    
        for i in range(N):
            img = worst_sample[i]
            img = img.transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')
    
            target_label_str = imagenet_labels[target_np[i]] if 0 <= target_np[i] < len(imagenet_labels) else f"Unknown:{target_np[i]}"
            pred_label_str = imagenet_labels[pred_np[i]] if 0 <= pred_np[i] < len(imagenet_labels) else f"Unknown:{pred_np[i]}"
    
            axes[i].set_title(f"{target_label_str}\n{pred_label_str}", fontsize=8)
    
        # 남은 subplot 비활성화
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
    
        # tight_layout()을 제거하거나, 먼저 호출 후 subplots_adjust 호출 가능
        # plt.tight_layout()  # 필요에 따라 주석처리
    
        # 세로 간격 조정: hspace 값을 줄여 세로 공간을 좁히기
        plt.subplots_adjust(wspace=0.3, hspace=0.2)
    
        plt.savefig(f"./results/worst_samples_{domain_id[0]}_{idx}.png")
        plt.close(fig)
    

    @staticmethod
    def configure_model(model: nn.Module,
                        device: str,
                        lora_alpha,
                        connection_type: str,
                        adapt_layers: List[str]) -> nn.Module:
        model = LoRA(model, lora_alpha, connection_type, adapt_layers)
        model.train()
        model = model.to(device)
        model.requires_grad_(False)
        model.set_trainable_param()
        return model

    @staticmethod
    def collect_params(model: nn.Module):
        params = []
        names = []
        for np, p in model.lora_layers.named_parameters():
            params.append(p)
            names.append(np)
        return params, names



class LoRA(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 lora_alpha: float,
                 connection_type: str,
                 adapt_layers: List[str]
                 ) -> None:
        super(LoRA, self).__init__()
        self.lora_alpha = lora_alpha
        self.connection_type = getattr(nn, connection_type)

        self.split_layers(model)
        self.lora_layers = nn.ModuleDict()
        self.global_pool = model.global_pool
        self.fc = model.fc
        
        for layer_idx in adapt_layers:
            original_layer = model.encoder[layer_idx]
            first_conv = self.get_first_conv_layer(original_layer)
            last_conv = self.find_last_conv_layer(original_layer)
            in_channels = first_conv.in_channels
            out_channels = last_conv.out_channels
            
            if layer_idx == 4:
                stride = 1
            else:
                stride = 2
            self.lora_layers[f'layer{layer_idx}'] = self.connection_type(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.initialize_lora_parameters()

    def forward(self, x):
        pred = {}

        out = self.pre_layer(x)
        out = self.layer1(out) + self.lora_alpha * self.lora_layers['layer4'](out)
        out = self.layer2(out) + self.lora_alpha * self.lora_layers['layer5'](out)
        out = self.layer3(out) + self.lora_alpha * self.lora_layers['layer6'](out)
        out = self.layer4(out) + self.lora_alpha * self.lora_layers['layer7'](out)
        
        pred['last_hidden_state'] = self.post_layer(out)
        pred['logits'] = self.fc(self.global_pool(out))
        return pred

    def split_layers(self, model: nn.Module) -> None:
        self.pre_layer = nn.Sequential()
        self.post_layer = nn.Sequential()
    
        # Flags to track position in the model
        in_pre_layer = True
        for name, module in model.encoder.named_children():
            if name == '4':
                in_pre_layer = False
                self.layer1 = module
            elif name == '5':
                self.layer2 = module
            elif name == '6':
                self.layer3 = module
            elif name == '7':
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