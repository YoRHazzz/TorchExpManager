import datetime
import os.path
import random

import numpy as np
import pandas as pd
import torch
from scipy import interpolate
from torch import nn


def detach_if_requires_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor if it requires gradient."""
    if tensor.grad_fn:
        return tensor.detach()
    else:
        return tensor


def seed_everything(seed: int = 1234, determinstic: bool = True):
    # https://github.com/dmlc/dgl/issues/3054
    # try:
    #     import dgl
    #     dgl.random.seed(seed)
    # except ImportError as e:
    #     pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if determinstic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def interpolate_to_window_size(raw_data: pd.DataFrame, window_size):
    interpolated_data = []
    x1 = np.linspace(0, window_size - 1, len(raw_data))
    x_new = np.linspace(0, window_size - 1, window_size)
    for yi in range(raw_data.shape[1]):
        tck = interpolate.splrep(x1, raw_data.values[:, yi])
        a = interpolate.splev(x_new, tck)
        interpolated_data.append(a)
    interpolated_data = pd.DataFrame(np.array(interpolated_data).T)
    interpolated_data.columns = raw_data.columns
    return interpolated_data


def get_depth(model):
    if hasattr(model, "depth"):
        return model.depth
    children = list(model.children())
    if len(children) == 0:
        return isinstance(model, (nn.Linear, nn.Conv2d))
    else:
        return sum(get_depth(child) for child in children)


def format_time(seconds):
    time_str = str(datetime.timedelta(seconds=seconds))
    if '.' in time_str:
        time_str = time_str.split(".")[0]
    if ':' in time_str:
        time_str = ":".join([time_slice for time_slice in time_str.split(":") if time_slice != "0"])
    return time_str


def current_time_str(time_format=None):
    time_format = time_format or '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.now().strftime(time_format)


def save_model(model: nn.Module, model_dir, ckpt_file_name, **kwargs):
    ckpt_path = os.path.join(model_dir, ckpt_file_name)
    check_point = {
        'state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(check_point, ckpt_path)
    print(f"{current_time_str()} model saved to {ckpt_path}")


def load_model(model: nn.Module, model_dir, ckpt_file_name, device):
    ckpt_path = os.path.join(model_dir, ckpt_file_name)
    check_point = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(check_point['state_dict'])
    print(f"{current_time_str()} model load from {ckpt_path}")
    return check_point


def get_devices(gpu):
    if isinstance(gpu, int):
        gpu = [gpu]
    if gpu:
        devices = [f'cuda:{index}' for index in gpu if index < torch.cuda.device_count()]
    else:
        devices = ['cpu']
    return devices
