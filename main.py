from config.tent import cfgs
import trainer
import wandb

def initialization(cfgs):
    wandb.login() #이거는 이미 login 되어 있으면 빼도 괜찮은거 같음.

    wandb.init(
        project='multiTTA', # Never Revise
        config=cfgs, # Frequentry Revise
        tags=['tent-baseline', 'imagenet-c'] # Frequentry Revise
    )

def main(cfgs):
    initialization(cfgs)
    trainer.

if __name__ == "__main__":
    main(cfgs)