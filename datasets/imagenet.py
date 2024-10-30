from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImageNet
import os

class ImageNet(Dataset):
    name = 'imagenet'

    """
    domain_id = {'train': 0, 'validation': 1}
    """
    def __init__(self, root = './data'):
        self.source_domain = TorchImageNet(os.path.join(root, 'imagenet'), split='train')
        self.target_domain = TorchImageNet(os.path.join(root, 'imagenet'), split='val')

    def __len__(self) -> int:
        return len(self.source_domain) + len(self.target_domain)

    def __getitem__(self, idx):
        assert idx < len(self)
        
        if idx < len(self.source_domain):
            domain_id = 0
            return self.source_domain[idx], domain_id
        else:
            idx -= len(self.source_domain)
            domain_id = 1
            return self.target_domain[idx], domain_id