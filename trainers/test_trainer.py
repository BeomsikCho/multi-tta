import torch.nn as nn
from torch.optim.optimizer import Optimizer
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
import re

from .tent_trainer import TentTrainer
from utils.builder import Builder
from utils.metric import softmax_entropy, empirical_fisher_information
from utils.common import device_seperation

class TestTrainer(TentTrainer):
    name = 'TestTrainer'

    def train_step(self, model, dataloader, optimizer):
        first_device, _ = device_seperation(self.device)
        
        model = self.set_dropout_layer(model) # 수정한 내용!!!!!!!!!

        model = self.configure_model(model, first_device)
        params, _ = self.collect_params(model)
        optimizer = self.adapt_optimizer(params, optimizer)

        train_loss = 0
        total_sample = 0 

        optimizer.zero_grad()


        for (samples, target, domain_id) in tqdm(dataloader):
            samples, target = samples.to(first_device), target.to(first_device)
            
            pred, uncertainty = self.mc_dropout(model, samples)

            breakpoint()
            
            loss = softmax_entropy(pred['logits']).mean(0)
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
    def set_dropout_layer(model):
        was_data_parallel = False  # DataParallel 여부 추적

        # DataParallel 확인 및 원래 모델로 접근
        if isinstance(model, nn.DataParallel):
            print("Detected DataParallel model. Using model.module for modification.")
            model = model.module
            was_data_parallel = True  # DataParallel임을 기록

        for nm, m in model.named_modules():
            # 이름의 마지막 부분이 'attn' 또는 'mlp'인지 확인
            last_name = nm.split('.')[-1]
            if last_name == 'attn' or last_name == 'mlp':
                print(f"{nm}에 dropout 삽입! : class = {m}")

                new_module = nn.Sequential(
                    m,
                    nn.Dropout(p=0.0)
                )

                # 부모 모듈과 자식 모듈 분리
                if '.' in nm:
                    parent_name, child_name = nm.rsplit('.', 1)
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, child_name, new_module)
                else:
                    # 루트 모듈의 경우
                    setattr(model, nm, new_module)

        # 다시 DataParallel로 래핑
        if was_data_parallel:
            model = nn.DataParallel(model)
            print("Re-wrapped the model into DataParallel.")

        return model
    
    @staticmethod
    def mc_dropout(model, samples):
        mean = 0
        sq_mean = 0
        T = 32

        for t in range(T):
            pred = model(samples)
            mean += pred['logits'] / T
            sq_mean += (pred['logits'] ** 2) / T

        uncertainty = torch.sqrt(sq_mean - mean ** 2)  # Var(X) = E[X^2] - (E[X])^2
        return mean, uncertainty.mean(-1)
        