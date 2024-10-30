from torch.utils.data import DataLoader

class SequentialDomainDataLoader(DataLoader):
    def __init__(self, datasets, batch_size=32):
        self.loaders = []
        for dataset in datasets:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.loaders.append(dataloader)

    def __iter__(self):
        self.loader_iter = iter(self.loaders)
        self.current_loader = None
        return self

    def __next__(self):
        if self.current_loader is None:
            self.current_loader = next(self.loader_iter).__iter__()
        
        try:
            # 현재 DataLoader에서 데이터를 반환
            return next(self.current_loader)
        except StopIteration:
            # 현재 DataLoader가 끝났다면, 다음 DataLoader로 이동
            self.current_loader = next(self.loader_iter).__iter__()
            return next(self.current_loader)