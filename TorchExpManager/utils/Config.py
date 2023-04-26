import warnings
from typing import Dict

import yaml
from tabulate import tabulate


class Config:
    def __init__(self, path):
        self.path = path

        self.base_config = {}
        self.sub_config = {}
        for key, value in self.read_config().items():
            if isinstance(value, Dict):
                self.sub_config[key] = value
            else:
                self.base_config[key] = value

        self.current_sub_config_name = None
        self.current_config = self.base_config.copy()

    def set_sub_config(self, sub_config_name):
        if sub_config_name not in self.sub_config:
            warnings.warn(f"sub config '{sub_config_name}' not in {self.sub_config.keys()}")
            self.sub_config[sub_config_name] = {}
        self.current_sub_config_name = sub_config_name
        self.current_config = self.base_config.copy()
        self.current_config.update(self.sub_config.get(sub_config_name, {}))

    def __getitem__(self, key):
        return self.current_config[key]

    def __setitem__(self, key, value):
        self.current_config[key] = value
        if self.current_sub_config_name and key in self.sub_config[self.current_sub_config_name]:
            self.sub_config[self.current_sub_config_name][key] = value
        else:
            self.base_config[key] = value

    def get(self, key, value):
        return self.current_config.get(key, value)

    def __contains__(self, key):
        return key in self.current_config

    def keys(self):
        return self.current_config.keys()

    def values(self):
        return self.current_config.values()

    def items(self):
        return self.current_config.items()

    def show_status(self):
        print(tabulate([*self.current_config.items()], tablefmt='grid',
                       numalign='center', stralign='center',
                       showindex=False))

    def read_config(self):
        """"读取配置"""
        with open(self.path, "r", encoding="utf-8") as yaml_file:
            config = yaml.load(yaml_file.read(), Loader=yaml.CLoader)
        return config

    def update_config(self):
        """"更新配置"""
        with open(self.path, 'w', encoding="utf-8") as yaml_file:
            yaml.dump(self.base_config.copy().update(self.sub_config), yaml_file, indent=2, default_flow_style=False)
        return self
