from trainers import Trainer
import wandb

from utils import Builder
from utils import setup_cfgs, setup_deterministic

def initialization():
    setup_deterministic(seed=2024)
    cfgs = setup_cfgs()
    
    wandb.login()
    wandb.init(
        project='multiTTA', # Never Revised
        config=cfgs, # Frequentry Revised
        tags=['tent-baseline', 'imagenet-c'] # Frequentry Revised
    )
    return cfgs

def main():
    cfgs = initialization()
    builder = Builder(cfgs)

    if cfgs['mode'] == 'train' or cfgs['mode'] == 'all':
        trainer = builder.build_trainer(cfgs)
        result = trainer.train()
        wandb.log(result)

    if cfgs['mode'] == 'eval' or cfgs['mode'] == 'all':
        evaluator = builder.build_evaluator(cfgs)
        result = evaluator.validate()
        wandb.log(result)
    
if __name__ == "__main__":
    main()