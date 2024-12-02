import torch.nn as nn
from torch.optim.optimizer import Optimizer
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .tent_trainer import TentTrainer
from utils.builder import Builder
from utils.metric import softmax_entropy, empirical_fisher_information
from utils.common import device_seperation

class EataTrainer(TentTrainer):
    name = 'EataTrainer'

    def train_step(self, model, dataloader, optimizer):
        first_device, _ = device_seperation(self.device)
        
        model = self.configure_model(model, first_device)
        params, _ = self.collect_params(model)
        optimizer = self.adapt_optimizer(params, optimizer)

        # Eata의 Loss 설정 부분
        loss_fn = EataLoss(e_margin = self.cfgs['e_margin'],
                           fisher_alpha = self.cfgs['fisher_alpha'],
                           d_margin = self.cfgs['d_margin'])
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


class EataLoss(object):
    def __init__(self, e_margin: float, d_margin: float, fisher_alpha: float):
        self.e_margin = e_margin # EMA threshold
        self.d_margin = d_margin # EMA threshold
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