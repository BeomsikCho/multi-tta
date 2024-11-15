from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

class ImageNetA(ImageFolder):
    name = 'imagenet-a'
    
    def __init__(self,
                 path: str,
                 ):
        self.domain_id = self.name

        root = Path(path) / self.name
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),  # Convert the images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(root, transform)

    @classmethod
    def build(cls,
              path: str = './data/',
              **others):
        yield ImageNetA(path)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, target, self.domain_id