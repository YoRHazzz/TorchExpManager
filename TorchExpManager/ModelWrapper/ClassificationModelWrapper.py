from typing import Dict, Any

import pandas as pd
import torch
from torch import nn

from .BaseModelWrapper import BaseModelWrapper


class ClassificationModelWrapper(BaseModelWrapper):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.criterion = nn.CrossEntropyLoss()

    def loss_func(self, out, y, metric_result):
        y = y.to(out.device)
        metric_result['loss'] = self.criterion(out, y)

    @torch.no_grad()
    def num_correct_func(self, out, y, metric_result):
        y = y.to(out.device)
        metric_result['num_correct'] = torch.eq(out.argmax(dim=1), y).sum()

    @torch.no_grad()
    def accuracy_func(self, out, y, metric_result):
        if 'num_correct' not in metric_result:
            self.num_correct_func(out, y, metric_result)
        if 'num_samples' not in metric_result:
            self.num_samples_func(out, y, metric_result)
        metric_result['accuracy'] = metric_result['num_correct'] / metric_result['num_samples']

    def collect(self, epoch_information: pd.DataFrame, epoch_summary: Dict[str, Any]) -> Dict[str, Any]:
        if 'num_correct' in epoch_information:
            epoch_summary['total_correct'] = epoch_information['num_correct'].sum()
            epoch_summary['accuracy'] = epoch_summary['total_correct'] / epoch_summary['total_samples']
        return epoch_summary
