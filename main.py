import wandb

import trainers
from utils import setup_cfgs, setup_deterministic

def initialization():
    setup_deterministic(seed=2024)
    cfgs = setup_cfgs()
    return cfgs

def main():
    cfgs = initialization()
    
    if cfgs['mode'] == 'train' or cfgs['mode'] == 'all':
        trainer_cls = getattr(trainers, cfgs['trainer']['name'])
        trainer = trainer_cls(cfgs)
        trainer.train()

    if cfgs['mode'] == 'eval' or cfgs['mode'] == 'all':
        # evaluator_cls = getattr(evaluators, cfgs['evaluator']['name'])
        # evaluator(cfgs) = evaluator_cls(cfgs)
        # result = evaluator.eval()
        # wandb.log(result)
        pass
    
if __name__ == "__main__":
    main()

