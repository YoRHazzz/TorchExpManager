import inspect
from typing import Dict, Callable, Any

import pandas as pd
import torch
from torch import nn


class BaseModelWrapper:
    def __init__(self, model: nn.Module, device):
        self.model = model.to(device)
        self.metric2func: Dict[str, Callable] = {}
        self.device = device
        for method_name, method_func in inspect.getmembers(self, predicate=inspect.ismethod):
            if method_name[-5:] == "_func":
                self.metric2func[method_name[:-5]] = method_func

    def loss_func(self, out, y, metric_result):
        raise NotImplementedError(self, "loss function")

    @torch.no_grad()
    def num_samples_func(self, out, y, metric_result):
        metric_result['num_samples'] = y.size(0)

    def __call__(self, x):
        x = x.to(self.device)
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def collect(self, epoch_information: pd.DataFrame, epoch_summary: Dict[str, Any]):
        raise NotImplementedError(self, "collect")

    def base_collect(self, epoch_information: pd.DataFrame) -> Dict[str, Any]:
        epoch_summary = {'total_samples': epoch_information['num_samples'].sum(),
                         'epoch_time': epoch_information['iter_time'].sum()}
        if 'loss' in epoch_information:
            epoch_summary['loss_mean'] = epoch_information['loss'].mean()
            epoch_summary['loss_std'] = epoch_information['loss'].std()
        return self.collect(epoch_information, epoch_summary)
