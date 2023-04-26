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
```

## How to use

1. 直接使用/继承 TorchExpManager.ModelWrapper。继承时重点在于实现自定义的xxx_func来实现xxx指标。
   目前ClassificationModelWrapper默认提供accuracy、num_correct指标的实现。

```python
import torch
from TorchExpManager.ModelWrapper import BaseModelWrapper
from typing import Dict, Any


class MyModelWrapper(BaseModelWrapper):
    def __init__(self, model, device):
        super().__init__(model, device)

    def loss_xxx_func(self, out, y, metric_result: Dict[str, Any]):
        # out: return value of self.model.forward(dataloader.iter_data['x'])
        # y: dataloader.iter_data['y']
        metric_result['loss_xxx'] = 123

    @torch.no_grad()
    def abc_func(self, out, y, metric_result: Dict[str, Any]):
        metric_result['abc'] = '456'
```

2. 直接使用/继承 TorchExpManager.DataloaderWrapper。继承时重点在于重写split_iter_data。
   iter_data['x']会当作样本传入model的forward函数.iter_data['y']会当作标签传入指标函数。

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
    valid_dataloader=test_dataloader,
    test_dataloader=test_dataloader,
    config=config,
    optimizer=optimizer,
    train_metrics={'loss', 'abc'},  # 指定训练时计算loss abc这两个指标
    valid_metrics={'abc'},  # 同上
    test_metrics={'loss_xxx'},  # 同上
    only_test=args.test,
    save_metric='loss_xxx',
    save_check_op='>',
    exp_name=args.exp_name
)
exp_manager = TorchExpManager(**kwargs)
exp_manager.run()
```

默认会将结果输出到logs/expN