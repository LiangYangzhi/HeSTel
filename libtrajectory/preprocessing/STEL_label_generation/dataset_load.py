import datetime
import os

import pandas as pd

from libtrajectory.utils.time_utils import parse_time, datetime_to_mktime


class DataLoader(object):
    def __init__(self, start_time, end_time):
        self.data = None
        self.start_time = parse_time(start_time) if isinstance(start_time, dict) else start_time
        self.end_time = parse_time(end_time) if isinstance(end_time, dict) else end_time
        self.device = None
        self.coverage = None
        self.ignore_error = None
        self.rename = None
        self.city = None
        self.columns = None
        self.subset = None

    def get_data(self, subset, columns, city=None, rename=None, ignore_error=True,
                 coverage=None, device_file=None):
        self.subset = subset
        self.columns = columns
        self.city = city
        self.rename = rename
        self.ignore_error = ignore_error
        self.coverage = coverage
        self.device = self._create_device(device_file)

        print(f"-{self.subset}")
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

    def _load_csv(self, path):
        if not os.path.exists(path):
            print(f"path: {path} is non-existent")
            if self.ignore_error:
                print("skip the path")
                return None
            else:
                raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        return df

    def _operate_column(self, df):
        if self.coverage:
            df = df.merge(self.device, how='left')
        if self.rename:
            df.rename(columns=self.rename, inplace=True)
        if self.columns:
            col = list(self.columns.values())
            df = df[col]
        return df

    def _astype(self):
        for col in ['user', 'device']:
            if col in self.columns:
                self.data[self.columns[col]] = self.data[self.columns[col]].astype('str')
        for col in ['user', 'longitude', 'latitude', 'device', 'coverage']:
            if col in self.columns:
                self.data[self.columns[col]] = self.data[self.columns[col]].astype('category')
