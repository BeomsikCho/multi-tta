from torch.utils.data import Dataset

from .imagenetA import ImageNetA
from .imagenetC import ImageNetC
from .imagenetV2 import ImageNetV2
from .imagenetSketch import ImageNetSketch
from .imagnetR import ImageNetR

from typing import Optional

class ImageNetAll(Dataset):
    name = 'imagenet-all'
    data_cls = [ImageNetA, ImageNetC, ImageNetV2, ImageNetSketch, ImageNetR]

    @classmethod
    def build(cls,
              path: str = './data/',
              data_cls: Optional[list] = None,
              **others):
        if data_cls is None:
            data_cls = cls.data_cls

        for cur_dataset in data_cls:
            for dataset in cur_dataset.build(path=path):
                yield dataset


if __name__ == "__main__":
    for data in ImageNetAll.build():
        print(f"Build {data.name}")

