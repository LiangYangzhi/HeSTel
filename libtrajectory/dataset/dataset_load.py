"""

"""
import datetime
import os

import pandas as pd

from libtrajectory.utils.time_utils import parse_time
from libtrajectory.dataset.abstract_dataset import AbstractDataset




class DataLoader(AbstractDataset):
    """
    Load dataset
    """

    def __init__(self):
        super().__init__()
        self.ignore_error = None
        self.columns = None
        self.rename = None

    def _read_csv(self, path) -> pd.DataFrame or None:
        """
        :param path: file path
        :return: pd.DataFrame or None
        """
        if not os.path.exists(path):
            print(f"path: {path} is non-existent")
            if self.ignore_error:
                print("skip the path")
                return None
            else:
                raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path)

        if self.rename:
            df.rename(columns=self.rename, inplace=True)
        if self.columns:
            df = df[self.columns]
        return df

    def reading(self, subset, columns, file=None, city=None, start_time=None, end_time=None,
                rename=None, ignore_error=True):
        """
        load face dataset
        :param subset: face/imsi/car/face/face_imsi_car
        :param columns: list or dict
            if columns == dict, columns = list(dict.values())
            if rename == None, columns = reading raw data need columns name
            if rename == dict, columns = Renamed column name
        :param file: str or list or None
            str: 文件名
            list: [文件名，文件名]  Todo
        :param city: str or None
            if subset == face/imsi/car, city = str
            if subset == face_imsi_car, city = None
        :param start_time: dict or None
            if subset == face/imsi/car, city = str,  example: {"year": 2022, "month": 2, "day": 21}
            if subset == face_imsi_car, city = None
        :param end_time: dict
            if subset == face/imsi/car, city = str, example: {"year": 2022, "month": 2, "day": 22}
            if subset == face_imsi_car, city = None
        :param rename: columns rename
        :param ignore_error: True or False
            True: ignore path file not exist.
        :return:
        """
        self.rename = rename
        self.columns = columns
        self.ignore_error = ignore_error
        if isinstance(self.columns, dict):
            self.columns = list(self.columns.values())

        if subset in ["face", "imsi", "car"]:
            start_time = parse_time(start_time)
            end_time = parse_time(end_time)
            days = (end_time - start_time).days + 1
            if days < 1:
                raise ValueError("time parameter error")

            dataset = []
            for day in range(days):
                loop_time = start_time + datetime.timedelta(days=day)
                loop_time_str = loop_time.strftime("%Y_%m_%d")
                print(f"--load {loop_time_str}")
                path = f"./libtrajectory/dataset/{subset}/{city}/{loop_time_str}.csv"
                df = self._read_csv(path)
                if isinstance(df, pd.DataFrame):
                    dataset.append(df)

            if len(dataset) > 0:
                df = pd.concat(dataset)
                return df
            else:
                return None

        elif subset == "face_imsi_car":
            path = f"./libtrajectory/dataset/{subset}/{file}.csv"
            df = self._read_csv(path)
            if isinstance(df, pd.DataFrame):
                return df
            return None

        else:
            raise f"subset = {subset} absent"


