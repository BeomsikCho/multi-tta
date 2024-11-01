from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import logging

class ImageNetR(ImageFolder):
    name = 'imagenet-r'
    
    def __init__(self,
                 path: str
                 ):
        root = Path(path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),  # Convert the images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(root, transform)

    @classmethod
    def build(cls,
              path: str,
              size: float):
        
        logging.info(f'build imagenet-r')
        yield ImageNetR(path)