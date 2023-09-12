"""
Data loader: loading data. The main functions include:

Data reading：It can read data from files and convert it into a DataFrame
Data access: It provides methods to access samples and labels in the dataset. Todo
"""

from abc import ABC


class AbstractDataset(ABC):
    """
    数据集类用于加载数据集
    """

    def reading(self, *args, **kwargs):
        """
        It can read data from files and convert it into a DataFrame
        :return DataFrame
        """
        pass

    def access(self, *args, **kwargs):
        """
        It provides methods to access samples and labels in the dataset
        """
        pass

