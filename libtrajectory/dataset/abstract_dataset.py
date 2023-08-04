from abc import ABC

import pandas as pd


class AbstractDataset(object, ABC):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        """
        返回数据集
        Returns: DataFrame
        """
        raise NotImplementedError("get_data not implemented")

    def deal_data(self):
        """
        对数据集做基础处理
        Returns: 处理好的数据集
        """
        raise NotImplementedError("deal_data not implemented")

    def save_data(self):
        """
        以csv格式保存DataFrame数据集
        """
        raise NotImplementedError("save_data not implemented")
