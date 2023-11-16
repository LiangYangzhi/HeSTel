import datetime
import os
import shutil
import time

import pandas as pd

from libtrajectory.utils.time_utils import parse_time


class Logs(object):

    def __init__(self):
        self._create_dir()

    def _create_dir(self):
        dir_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        path = f"./libtrajectory/logs/STEL_STIntersection/{dir_name}"
        os.makedirs(path)
        self.path = path

    def save_params(self):
        if os.path.exists('./STEL_STIntersection_run.logs'):
            shutil.copy('./STEL_STIntersection_run.logs', f"{self.path}/STEL_STIntersection_run.logs")

    def save_feature(self, data: pd.DataFrame, start_time, end_time):
        name = self._naming(start_time, end_time)
        path = f"{self.path}/feature_{name}.csv"
        data.to_csv(path)

    def _naming(self, start_time, end_time):
        if isinstance(start_time, dict):
            start_time = str(parse_time(start_time))
            end_time = str(parse_time(end_time))
        if isinstance(start_time, datetime.datetime):
            start_time = str(start_time)
            end_time = str(end_time)
        name = f"{start_time}_{end_time}"

        return name
