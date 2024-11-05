from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

class ImageNetC(ImageFolder):
    name = 'imagenet-c'

    corruptions = {
        'gaussian_noise': 0,
        'shot_noise': 1,
        'impulse_noise': 2,
        'defocus_blur' : 3, 
        'glass_blur' : 4, 
        'motion_blur' : 5, 
        'zoom_blur' : 6,
        'snow' : 7, 
        'frost' : 8, 
        'fog' : 9, 
        'brightness' : 10, 
        'contrast' : 11, 
        'elastic_transform' : 12, 
        'pixelate' : 13, 
        'jpeg_compression': 14
    }

    levels = [1, 2, 3, 4, 5]

    def __init__(self,
                 path: str,
                 corrupt: str,
                 level: int
                 ):
        self.domain_id = corrupt
                 
        self.name = f"{self.name}/{corrupt}/{level}"
        root = Path(path) / self.name /corrupt / str(level)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),  # Convert the images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(root, transform)

    @classmethod
    def build(cls,
              path: str,
              corruptions = None,
              levels = None,
              **others):
        assert set(corruptions).issubset(cls.corruptions.keys())

        if corruptions == None:
            corruptions = cls.corruptions.keys()
        if levels == None:
            levels = [5]

        for level in levels:
            for corrupt in corruptions:
                yield ImageNetC(path, corrupt, level)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, target, self.domain_id
    
    