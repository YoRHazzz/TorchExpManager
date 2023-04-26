# TorchExpManager

PyTorch项目脚手架，类似pytorch_lightning，目标是自动训练、验证、测试模型，可自定义各种指标并将结果保存为csv文件。

A manager for training, validation and testing a PyTorch model

## Requirements

```
tabulate
numpy
pandas
scipy
pyyaml
torch
torchvision
```

## Example

```shell
python AlexNet.py
# python AlexNet.py --test --exp_name exp1
```

## Result
1. 生成训练中每个epoch和每个iter的信息表格，包括运行时间和指定的指标。
```
logs
└── exp1
     ├── epoch_details
     │         ├── epoch001_train.csv # epoch001 训练的iter信息
     │         ├── epoch002_train.csv # epoch002 训练的iter信息
     │         ├── epoch002_valid.csv # epoch002 验证的iter信息
     │         ├── epoch002_test.csv  # epoch002 测试的iter信息
     ├── saved_model
     │         └── model_best.pt # 保存的模型
     ├── summary_table.csv       # 下面三个表的合并表
     ├── test_summary_table.csv  # 测试的总结信息
     ├── train_summary_table.csv # 训练的总结信息，包括所有epoch
     └── valid_summary_table.csv # 验证的总结信息，包括所有epoch
```
2. epoch进度条，以表格形式格式化输出指定的指标，并且自动评估预期结束时间。
```
Epoch 024/100 train: 100%|##########| 60000/60000 [00:31<00:00, 1880.39samples/s, loss_mean=0.0677, accuracy=0.975]
Epoch 024/100 valid: 100%|##########| 10000/10000 [00:04<00:00, 2328.44samples/s, loss_mean=0.332, accuracy=0.925]
2023-04-26 18:59:44 model saved to logs/tmp/saved_model/model_best.pt
+---------+---------+-------------+------------+-----------------+-----------------+------------+--------------+
|  epoch  |  stage  |  loss_mean  |  loss_std  |  total_samples  |  total_correct  |  accuracy  |  epoch_time  |
|---------+---------+-------------+------------+-----------------+-----------------+------------+--------------|
|   24    |  train  |  0.0677369  | 0.0252625  |      60000      |      58500      |   0.975    |   31.7553    |
|   24    |  valid  |  0.331805   | 0.0846502  |      10000      |      9253       |   0.9253   |   4.12765    |
+---------+---------+-------------+------------+-----------------+-----------------+------------+--------------+
Time: 00:35 -> 13:13/55:44 | Expected end Time: 2023-04-26 19:42:15
Best accuracy: 0.9253 (epoch 24) | Early stop count: 0/5
```
3. 按照指定的某个指标实现early stop，并且保存模型。
```
Sanity Check: save metric = 'accuracy'
Sanity Check: save check op = '>' [current accuracy > best accuracy]
Sanity Check: early stop threshold = 5
```

## How to use

1. 直接使用/继承 TorchExpManager.ModelWrapper。

继承时重点在于
   
- 自定义的xxx_func来实现xxx指标。目前ClassificationModelWrapper默认提供accuracy、num_correct指标的实现。
- 自定义collect：从每个iter收集信息生成当前epoch的summary

```python
import torch
from torch import nn
from TorchExpManager.ModelWrapper import BaseModelWrapper
from typing import Dict, Any
import pandas as pd

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
```

2. 直接使用/继承 TorchExpManager.DataloaderWrapper。

继承时重点在于
- 重写split_iter_data。iter_data['x']会当作样本传入model的forward函数.iter_data['y']会当作标签传入指标函数。

```python
from torch.utils.data import DataLoader
from typing import Dict, Any


class BaseDataLoaderWrapper:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter_data: Dict[str, Any] = {'x': None, 'y': None}

    def split_iter_data(self, iter_data):
        self.iter_data['x'], self.iter_data['y'] = iter_data
        # 'x': samples passed to the model's forward method. | model(x)
        # 'y': labels passed to xxx_func. | xxx_func(out, y, metric_result)

    def __iter__(self):
        for iter_data in self.dataloader:
            self.split_iter_data(iter_data)
            yield self.iter_data
```

3. 开始实验。Config文件是yaml配置文件，默认通过'num_epochs'保存训练的总epoch数，'eval_interval'保存每几个epoch进行一次验证。

注意：可以通过xxx_metrics指定需要计算什么参数

```python
import torch
from TorchExpManager import TorchExpManager
from TorchExpManager.utils import Config

...
config = Config('default.yaml')
train_dataloader = BaseDataLoaderWrapper(train_dataloader)
...
device = torch.device(...)
model = ...
model_wrapper = MyModelWrapper(model, device)
optimizer = ...

kwargs = dict(
    model_wrapper=model_wrapper,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    test_dataloader=test_dataloader,
    config=config,
    optimizer=optimizer,
    train_metrics={'loss', 'abc'},  # 指定训练时计算loss abc这两个指标
    valid_metrics={'abc'},  # 同上
    test_metrics={'loss_xxx'},  # 同上
    only_test=args.test,
    save_metric='loss_xxx',
    save_check_op='>',
    exp_name=args.exp_name,
    early_stop_threshold=5,
)
exp_manager = TorchExpManager(**kwargs)
exp_manager.run()
```

默认会将结果输出到logs/expN