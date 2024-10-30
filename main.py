from trainers import Trainer
import wandb

from utils import Builder
from utils import setup_deterministic

def initialization():
    from config.tent import cfgs # Frequentry Revised
    wandb.login() #이거는 이미 login 되어 있으면 빼도 괜찮은거 같음.
    wandb.init(
        project='multiTTA', # Never Revised
        config=cfgs, # Frequentry Revised
        tags=['tent-baseline', 'imagenet-c'] # Frequentry Revised
    )
    
    setup_deterministic(seed=2024)
    return cfgs

def main():
    cfgs = initialization()
    builder = Builder(cfgs)

    trainer = builder.build_trainer(cfgs)
    result = trainer.train()
    wandb.log(result)

if __name__ == "__main__":
    main()