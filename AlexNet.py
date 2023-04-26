import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from TorchExpManager import TorchExpManager
from TorchExpManager.DataLoaderWrapper import BaseDataLoaderWrapper
from TorchExpManager.ModelWrapper import ClassificationModelWrapper
from TorchExpManager.utils import seed_everything, Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


def main():
    parser = argparse.ArgumentParser(description=f'{NET_NAME}')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='path to configuration file')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the saved model')
    parser.add_argument('--exp_name', type=str, default=None, help='directory to save summary and model')
    args = parser.parse_args()
    config = Config(args.config)

    if 'seed' in config:
        seed_everything(config['seed'], determinstic=True)

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"Sanity Check: use gpu = {devices}")

    resize = config.get('resize', 224)
    transforms_list = [transforms.ToTensor()]
    if resize > 0:
        transforms_list.append(transforms.Resize(resize, antialias=False))
    normal_augs = transforms.Compose(transforms_list)
    train_dataset = datasets.FashionMNIST(root='data/', train=True, transform=normal_augs, download=True)
    test_dataset = datasets.FashionMNIST(root='data/', train=False, transform=normal_augs, download=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=config.get('num_workers', 0))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['test_batch_size'], shuffle=False,
                                 num_workers=config.get('num_workers', 0))

    model = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device("cuda:1")

    train_dataloader, test_dataloader = BaseDataLoaderWrapper(train_dataloader), BaseDataLoaderWrapper(test_dataloader)
    model_wrapper = ClassificationModelWrapper(model, device)

    kwargs = dict(
        model_wrapper=model_wrapper,
        train_dataloader=train_dataloader,
        valid_dataloader=test_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        optimizer=optimizer,
        train_metrics={'accuracy'},
        valid_metrics={'accuracy', 'loss'},
        test_metrics={'accuracy'},
        only_test=args.test,
        save_metric='accuracy',
        save_check_op='>',
        exp_name=args.exp_name
    )
    exp_manager = TorchExpManager(**kwargs)
    exp_manager.run()


if __name__ == "__main__":
    main()
