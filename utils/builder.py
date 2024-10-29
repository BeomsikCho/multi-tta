from typing import Optional

class Builder(object):
    def __init__(self, cfgs: Optional[dict] = None):
        if cfgs:
            self.cfgs = cfgs

    def build_model(self, model: str, pretrained: bool = True):
        pass

    def build_dataloder(self, dataset: str):
        pass

    def build_optimizer(self, optimizer: str):
        pass

    def build_scheduler(self, scheduler: str):
        pass


if __name__ == "__main__":
    # Test the operation of Builder class
    builder = Builder()

