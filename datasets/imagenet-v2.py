from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import logging

class ImageNetV2(ImageFolder):
    name = 'imagenet-v2'

    data_types = ['imagenetv2-matched-frequency-format-val',
                 'imagenetv2-threshold0.7-format-val',
                 'imagenetv2-top-images-format-val']
    
    def __init__(self,
                 path: str,
                 data_type: str,
                 ):
        self.name = f"{self.name}/{data_type}"
        root = Path(path) / data_type
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),  # Convert the images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(root, transform)

    @classmethod
    def build(cls,
              path: str,
              size: float,
              data_types = None
              ):
        
        if data_types == None:
            data_types = cls.data_types

        for data_type in data_types:
            logging.info(f'build imagenet-c / data_type: {data_type}')
            yield ImageNetV2(path, data_type)