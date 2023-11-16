import datetime
import os
import pandas as pd

from libtrajectory.utils.time_utils import parse_time, datetime_to_mktime


class DataLoader(object):
    """
    Load dataset
    """

    def __init__(self):
        super().__init__()
        self.filter = None
        self.data = None
        self.label_file = None
        self.end_time = None
        self.start_time = None
        self.city = None
        self.subset = None
        self.device = None
        self.coverage = None
        self.ignore_error = None
        self.columns = None
        self.rename = None

    def get_data(self, subset, columns, city=None, start_time=None, end_time=None,
                 rename=None, ignore_error=True, coverage=None, device_file=None, label_file=None, filter=None):
        """
        load face dataset
        :param subset: face/imsi/car
        :param columns: dict
            columns = Renamed column name
        :param city: str or None
            if subset == face or imsi or car, city = str
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
        :param coverage: None or list
            None: Do not generate coverage column
            list: generate coverage column
        :param device_file: None or str
            None: not load device file
            str: device file name
        :param label_file: None or str
            None: not load label file
            str: label file name
        :param filter: None or dict
            None: non filter
            dict: {column_name: [value, ...], ...}
        :return:
        """
        print(f"--load {subset}")
        self.subset = subset
        self.columns = columns
        self.city = city
        self.start_time = start_time if start_time is None else parse_time(start_time)
        self.end_time = end_time if end_time is None else parse_time(end_time)
        self.rename = rename
        self.ignore_error = ignore_error
        self.coverage = coverage
        self.device = self._create_device(device_file)
        self.label_file = label_file
        self.filter = filter

        if self.label_file:
            path = f"./libtrajectory/dataset/{self.subset}/{self.label_file}.csv"
            df = self._load_csv(path)
            print(df.info())

            if isinstance(df, pd.DataFrame):
                df = self._operate_column(df)

            self.data = df
            for col in self.columns.values():
                self.data[col] = self.data[col].astype('str')
            self._filter()
            return self.data
        else:
            days = (self.end_time - self.start_time).days + 1
            if days < 1:
                raise ValueError("time parameter error")

            dataset = []
            for day in range(days):
                loop_time = self.start_time + datetime.timedelta(days=day)
                loop_time_str = loop_time.strftime("%Y_%m_%d")
                print(f"--load {loop_time_str}")
                path = f"./libtrajectory/dataset/{self.subset}/{self.city}/{loop_time_str}.csv"
                df = self._load_csv(path)

                if isinstance(df, pd.DataFrame):
                    df = self._operate_column(df)
                dataset.append(df)

            if len(dataset) == 0:
                raise ValueError("dataset is empty")

            self.data = pd.concat(dataset)
            self._astype()
            self._filter()
            return self.data

    def _create_device(self, device_file):
        if isinstance(self.coverage, list):
            device = pd.read_csv(f"./libtrajectory/dataset/{self.subset}/{self.city}/{device_file}.csv")
            columns = ["deviceId", "deviceModel"]
            device = device[columns]

            if len(self.coverage) == 1:
                device["coverage"] = self.coverage[0]
            if len(self.coverage) == 2:
                device["coverage"] = device["deviceModel"].map(
                    lambda s: self.coverage[1] if "BIG" in str(s) else self.coverage[0])

            device["deviceId"] = device["deviceId"].astype('category')
            return device
        return None

    def _operate_column(self, df):
        if self.coverage:
            df = df.merge(self.device, how='left')
        if self.rename:
            df.rename(columns=self.rename, inplace=True)
        if self.columns:
            col = list(self.columns.values())
            df = df[col]
            df = df.dropna()
        return df

    def _load_csv(self, path):
        if not os.path.exists(path):
            print(f"path: {path} is non-existent")
            if self.ignore_error:
                print("skip the path")
                return None
            else:
                raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path)
        return df

    def _astype(self):
        for col in ['user', 'device']:
            if col in self.columns:
                self.data[self.columns[col]] = self.data[self.columns[col]].astype('str')
        for col in ['user', 'longitude', 'latitude', 'device', 'coverage']:
            if col in self.columns:
                self.data[self.columns[col]] = self.data[self.columns[col]].astype('category')

    def _filter(self):
        if isinstance(self.filter, dict):
            for col, value in self.filter.items():
                self.data = self.data.query(f"{self.columns[col]} in {value}")
