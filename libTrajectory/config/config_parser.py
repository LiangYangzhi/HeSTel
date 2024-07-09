import os
import json


def parse_config(path):
    if os.path.exists(f'./libTrajectory/config/{path}.json'):
        with open(f'./libTrajectory/config/{path}.json', 'r') as f:
            config = json.load(f)
    else:
        print(f'./libTrajectory/config/{path}.json')
        raise FileNotFoundError(f'Config file {path}.json is not found.')
    return config


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, saved_model=True, other_args=None, hyper_config_dict=None):
        """
        :preprocessing
        :param saved_model:
        :param other_args: dict
            超参数调整时传入的待调整的参数，优先级低于命令行参数
        :param hyper_config_dict: dict
            通过命令行进行传参的参数，优先级最高
        """
        self.config = {}
        if hyper_config_dict is not None:
            for key in hyper_config_dict:
                if key not in self.config:
                    self.config[key] = hyper_config_dict[key]
        if other_args is not None:
            for key in other_args:
                if key not in self.config:
                    self.config[key] = other_args[key]
        if "saved_model" not in self.config:
            self.config["saved_model"] = saved_model

    def add_dataset_config(self, config_file):
        dic = parse_config(config_file)
        if "dataset" not in self.config:
            self.config["dataset"] = dic
        else:
            for key in dic:
                if key not in self.config['dataset']:
                    self.config["dataset"][key] = dic[key]

    def add_preprocessing_config(self, config_file):
        dic = parse_config(config_file)
        if "preprocessing" not in self.config:
            self.config["preprocessing"] = dic
        else:
            for key in dic:
                if key not in self.config['preprocessing']:
                    self.config["preprocessing"][key] = dic[key]

    def add_model_config(self, config_file):
        dic = parse_config(config_file)
        if "model" not in self.config:
            self.config["model"] = dic
        else:
            for key in dic:
                if key not in self.config['model']:
                    self.config["model"][key] = dic[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config
