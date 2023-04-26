from typing import Dict, Any

from torch.utils.data import DataLoader


class BaseDataLoaderWrapper:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter_data: Dict[str, Any] = {'x': None, 'y': None}

    def split_iter_data(self, iter_data):
        self.iter_data['x'], self.iter_data['y'] = iter_data

    def __iter__(self):
        for iter_data in self.dataloader:
            self.split_iter_data(iter_data)
            yield self.iter_data

    @property
    def num_samples(self):
        return len(self.dataloader.dataset)
