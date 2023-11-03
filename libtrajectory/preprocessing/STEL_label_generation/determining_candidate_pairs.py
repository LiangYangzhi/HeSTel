import gc

import numpy as np
import pandas as pd
from pandarallel import pandarallel


class DeterminingCandidatePairs(object):
    def __init__(self, data1: pd.DataFrame, col1: dict, data2: pd.DataFrame, col2: dict,
                 batch_size: int, front_time: int, back_time: int, device_distance):
        """
        :param data1: pd.DataFrame
            trajectory data, data1在data2内寻找关系对
        :param col1: dict
            The columns names corresponding to trajectory data1
            {user: data1 column name,
            time: data1 column name,
            longitude: data1 column name,
            latitude: data1 column name,
            device: data1 column name, ...}
        :param data2: pd.DataFrame
            trajectory data
        :param col2: dict
            The columns names corresponding to trajectory data2
            {user: data2 column name,
            time: data2 column name,
             ...}
         :param batch_size: batch size max number
         :param front_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param back_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        """
        self.data1 = data1
        self.col1 = col1
        self.data2 = data2
        self.col2 = col2
        self.front_time = front_time
        self.back_time = back_time
        self.batch_size = batch_size

        self.device_distance = device_distance
        self.num_top = None
        self.place_top = None

    def sti_place_num(self, segment: pd.DataFrame, col_segment: dict,
                      place_top=None, num_top=None):
        self.place_top = place_top
        self.num_top = num_top
        self._set_index()

        segment.reset_index(drop=True, inplace=True)
        if segment.shape[0] > self.batch_size:
            list_df = [pd.DataFrame(sub_df) for sub_df in np.array_split(segment, len(segment) // self.batch_size)]
            for i, sub_df in enumerate(list_df):
                print(f"{i}/{len(list_df)}")
                pandarallel.initialize(progress_bar=False)
                sub_df['feature'] = sub_df.parallel_apply(
                    lambda row: self._candidate_user(
                        row[col_segment['user1']], row[col_segment['start_time']], row[col_segment['end_time']]
                    ), axis=1)
            segment = pd.concat(list_df)
        else:
            segment['feature'] = segment.parallel_apply(
                lambda row: self._candidate_user(
                    row[col_segment['user1']], row[col_segment['start_time']], row[col_segment['end_time']]
                ), axis=1)

        segment = segment.dropna()
        segment.reset_index(drop=True, inplace=True)
        feature = segment.explode("feature")
        feature.reset_index(drop=True, inplace=True)
        feature = pd.concat([feature, feature['feature'].apply(pd.Series)],
                            axis=1).drop('feature', axis=1)

        feature_col = {"user1": self.col1['user'],
                       "user2": self.col2['user']}
        lis = [c for c in feature.columns if c not in feature_col]
        feature_col['feature'] = lis

        return feature, feature_col

    def _set_index(self):
        need_col1 = [self.col1['user'], self.col1['time'], self.col1['device']]
        self.data1 = self.data1[need_col1]
        need_col2 = [self.col2['user'], self.col2['time'], self.col2['device']]
        self.data2 = self.data2[need_col2]

        index1 = [self.col1['user']]
        self.data1.set_index(index1, drop=True, inplace=True)

        index2 = [self.col2['device']]
        self.data2.set_index(index2, drop=True, inplace=True)

        index_col = [self.col1['device']]  # 决定取值的顺序，index通过位置获取values
        self.device_distance.set_index(index_col, drop=True, inplace=True)

    def _sti_device(self, time1, device1):
        # [0]: user, [1]: time,  [2]: device
        sti_start_time = int(time1 - self.back_time)
        sti_end_time = int(time1 + self.front_time)
        device2 = self.device_distance.query(
            f"{self.col1['device']} == '{device1}'")[self.col2['device']].unique().tolist()
        if not device2:
            return None
        sti_user = self._candidate_data.query(
            f"({self.col2['device']} in {device2}) & "
            f"({sti_end_time} >= {self.col2['time']}) & ({self.col2['time']} >= {sti_start_time})"
        )
        if sti_user.shape[0]:
            return sti_user[self.col2['user']].unique().tolist()
        return None

    def _candidate_user(self, user1, start_time, end_time):
        data1 = self.data1.query(f"{self.col1['user']} == @user1").query(
            f"({end_time} >= {self.col1['time']} >= {start_time})"
        )
        device1 = data1[self.col1['device']].unique().tolist()
        device2 = self.device_distance.query(f"{self.col1['device']} in {device1}")[
            self.col2['device']].unique().tolist()
        sti_start_time = start_time - self.back_time
        sti_end_time = end_time + self.front_time
        self._candidate_data = self.data2.query(f"{self.col2['device']} in {device2}").query(
            f"({sti_end_time} >= {self.col2['time']} >= {sti_start_time})"
        )

        data1[self.col2['user']] = data1.apply(lambda row: self._sti_device(
            row[self.col1['time']], row[self.col1['device']]), axis=1)
        data1 = data1.dropna()
        if data1.shape[0] == 0:
            return None
        data1 = data1.explode(self.col2['user'])
        data1 = data1.reset_index()
        sti = data1.groupby(self.col2['user']).agg({self.col1['device']: list})  # sti: spatiotemporal intersection

        num_name = "num"  # 时空交集的次数
        device_name = "device"  # 时空交集的设备
        device_num_name = "device_set_num"  # 时空交集的设备去重后个数
        sti[num_name] = sti[self.col1['device']].map(lambda row: len(row))
        sti[device_name] = sti[self.col1['device']].map(lambda row: set(row))
        sti[device_num_name] = sti[self.col1['device']].map(lambda row: len(set(row)))

        if self.num_top is None:
            num_user2 = sti.sort_values(by=num_name, ascending=False).index.tolist()
        else:
            num_user2 = sti.sort_values(by=num_name, ascending=False).head(self.num_top).index.tolist()
        if self.place_top is None:
            device_user2 = sti.sort_values(by=device_num_name, ascending=False).index.tolist()
        else:
            device_user2 = sti.sort_values(
                by=device_num_name, ascending=False).head(self.place_top).index.tolist()

        candidate_user2 = list(set(num_user2 + device_user2))
        if candidate_user2.__len__() == 0:
            return None
        sti = sti.query(f"{self.col2['user']} in {candidate_user2}")
        feature = [{self.col2['user']: user2,
                    num_name: sti.query(f"{self.col2['user']} == @user2")[num_name].values[0],
                    device_name: sti.query(f"{self.col2['user']} == @user2")[device_name].values[0]}
                   for user2 in candidate_user2]
        return feature
