import wandb

import trainers
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
    
    if cfgs['mode'] == 'train' or cfgs['mode'] == 'all':
        trainer = getattr(trainers, cfgs['trainer'])
        result = trainer.train()
        wandb.log(result)

    if cfgs['mode'] == 'eval' or cfgs['mode'] == 'all':
        evaluator = getattr(evaluator, cfgs['evaluator'])
        result = evaluator.eval()
        wandb.log(result)
    
if __name__ == "__main__":
    main()


