import pandas as pd
from progress.bar import Bar

from libtrajectory.utils.coordinate import device_distance
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


class DeterminingCandidatePairs(object):
    def __init__(self, data1=None, col1=None, data2=None, col2=None):
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
        """
        self.data1 = data1
        self.col1 = col1
        self.data2 = data2
        self.col2 = col2
        self.front_time = None
        self.back_time = None
        self.space_distance = None
        self.into = None
        self.device_distance = None
        self.place_top = None
        self.num_top = None

    def _create_device_distance_table(self):
        device1_col = [self.col1['device'], self.col1['longitude'], self.col1['latitude']]
        device1 = self.data1[device1_col]
        device1 = device1.drop_duplicates(subset=device1_col)
        device2_col = [self.col2['device'], self.col2['longitude'], self.col2['latitude']]
        device2 = self.data2[device2_col]
        device2 = device2.drop_duplicates(subset=device2_col)
        distance_name = "distance"  # distance between devices
        df = device_distance(
            device1, device1_col, device2, device2_col, self.space_distance, distance_name
        )
        index_col = [self.col1['device']]  # 决定取值的顺序，index通过位置获取values
        df.set_index(index_col, drop=True, inplace=True)
        return df

    def _sti_device(self, device1, time, data2):
        sti_start_time = time - self.back_time
        sti_end_time = time + self.front_time
        device2 = self.device_distance[
            self.device_distance[self.col1['device']].isin([device1])][self.col2['device']].unique().tolist()
        if not device2:
            return None
        sti_user = data2[(data2[self.col2['device']].isin(device2)) &
                         (data2[self.col2['time']].isin(range(sti_start_time, sti_end_time + 1)))][
            self.col2['user']].unique().tolist()
        if not sti_user:
            return None
        return sti_user

    def _candidate_user(self, user1, start_time, end_time):
        data1: pd.DataFrame = self.data1[(self.data1[self.col1['user']].isin([user1])) &
                                         (self.data1[self.col1['time']].isin(range(start_time, end_time + 1)))]
        if self.into == "device":
            device1 = data1[self.col1['device']].unique().tolist()
            device2 = self.device_distance[
                self.device_distance[self.col1['device']].isin(device1)][self.col2['device']].unique().tolist()
            sti_start_time = start_time - self.back_time
            sti_end_time = end_time + self.front_time
            data2 = self.data2[(self.data2[self.col2['device']].isin(device2)) & (
                self.data2[self.col2['time']].isin(range(sti_start_time, sti_end_time + 1)))]
            data1[self.col2['user']] = data1.apply(
                lambda row: self._sti_device(row[self.col1['device']], row[self.col1['time']], data2), axis=1)
            data1 = data1.dropna()
            data1 = data1.explode(self.col2['user'])

            # The column names of the data1 are: [user, time, longitude, latitude, device, user2(sti)]
            # sti dataFrame: index is user2, column is data1 device
            sti = data1.groupby(self.col2['user']).agg({self.col1['device']: list})  # sti: spatiotemporal intersection
            # 时空交集的次数
            num_name = "num_lyz" if "num" in sti.columns else "num"
            # 时空交集的设备
            device_name = "device_lyz" if "device" in sti.columns else "device"
            sti[num_name] = sti[self.col1['device']].map(lambda row: len(row))
            sti[device_name] = sti[self.col1['device']].map(lambda row: len(set(row)))

            if self.num_top is None:
                num_user2 = sti.sort_values(by=num_name, ascending=False).index.tolist()
            else:
                num_user2 = sti.sort_values(by=num_name, ascending=False).head(self.num_top).index.tolist()
            if self.place_top is None:
                device_user2 = sti.sort_values(by=device_name, ascending=False).index.tolist()
            else:
                device_user2 = sti.sort_values(
                    by=device_name, ascending=False).head(self.place_top).index.tolist()

            candidate_user2 = list(set(num_user2 + device_user2))
            return candidate_user2

    def sti_place_num(self, segment: pd.DataFrame, col_segment: dict, into: str,
                      front_time: int, back_time: int, space_distance: int,
                      place_top=None, num_top=None):
        """
        通过时空交集(sti: spatio-temporal intersection)的场所数和时空交集的次数来确定候选关系对
        :param segment: pd.DataFrame
            根据data1轨迹数据进行切割的数据
        :param col_segment: dict
            The columns names corresponding to trajectory segment, is 
            {user1: segment column name, 
            segment: segment column name, 
            start_time: segment column name, 
            end_time: segment column name}
        :param into: str
            lon_lat: 时空交集所需距离直接通过经纬度计算距离, longitude and latitude  Todo
            device: 预先计算设备之间的经纬度, 时空交集所需距离通过设备
        :param front_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param back_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param space_distance: int, unit is meter
            时空交集的空间阈值, data1的轨迹点地理位置为p，在进行空间交集时，寻找data2在[p-space, p+space]距离内的轨迹点。
        :param place_top: None or int
            into=device时起作用
            None: 不进行过滤
            int: data1的user1与data2的user2产生时空交集user1的device去重后，根据device数量对user2进行排序，取前place_top的user2。
        :param num_top: None or int
            None: 不进行过滤
            int: data1的user1与data2的user2产生时空交集，根据时空交集的次数对user2进行排序，取前place_top的user2。
        :return: pd.DataFrame,  columns is [user1, user2, segment, start_time, end_time]
        """

        self.into = into
        if self.into == "device":
            self.space_distance = space_distance
            self.device_distance = self._create_device_distance_table()
            self.front_time = front_time
            self.back_time = back_time
            self.place_top = place_top
            self.num_top = num_top

            segment[self.col2['user']] = segment.parallel_apply(
                lambda row: self._candidate_user(
                    row[col_segment['user1']], row[col_segment['start_time']], row[col_segment['end_time']]), axis=1)
            pairs = segment.explode(self.col2['user'])
            pairs = pairs.dropna()
            # [user1, user2, segment, start_time, end_time]
            pairs_col = [col_segment['user1'], self.col2['user'],
                         col_segment['segment'], col_segment['start_time'], col_segment['end_time']]
            pairs = pairs[pairs_col]
            return pairs, pairs_col

    def _sti_device_index(self, time1, device1):
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

    def _candidate_user_index(self, user1, start_time, end_time):
        data1 = self.data1.query(f"{self.col1['user']} == @user1").query(
            f"({end_time} >= {self.col1['time']} >= {start_time})"
        )
        if self.into == "device":
            device1 = data1[self.col1['device']].unique().tolist()
            device2 = self.device_distance.query(f"{self.col1['device']} in {device1}")[
                self.col2['device']].unique().tolist()
            sti_start_time = start_time - self.back_time
            sti_end_time = end_time + self.front_time
            self._candidate_data = self.data2.query(f"{self.col2['device']} in {device2}").query(
                f"({sti_end_time} >= {self.col2['time']} >= {sti_start_time})"
            )
            data1[self.col2['user']] = data1.apply(lambda row: self._sti_device_index(
                row[self.col1['time']], row[self.col1['device']]), axis=1)
            data1 = data1.dropna()
            if data1.shape[0] == 0:
                return None
            data1 = data1.explode(self.col2['user'])
            data1 = data1.reset_index()
            sti = data1.groupby(self.col2['user']).agg({self.col1['device']: list})  # sti: spatiotemporal intersection

            num_name = "num"  # 时空交集的次数
            device_name = "device"  # 时空交集的设备
            sti[num_name] = sti[self.col1['device']].map(lambda row: len(row))
            sti[device_name] = sti[self.col1['device']].map(lambda row: len(set(row)))

            if self.num_top is None:
                num_user2 = sti.sort_values(by=num_name, ascending=False).index.tolist()
            else:
                num_user2 = sti.sort_values(by=num_name, ascending=False).head(self.num_top).index.tolist()
            if self.place_top is None:
                device_user2 = sti.sort_values(by=device_name, ascending=False).index.tolist()
            else:
                device_user2 = sti.sort_values(
                    by=device_name, ascending=False).head(self.place_top).index.tolist()

            candidate_user2 = list(set(num_user2 + device_user2))
            if candidate_user2.__len__() == 0:
                return None
            return candidate_user2

    def _create_index(self):
        index1 = [self.col1['user']]
        self.data1.set_index(index1, drop=True, inplace=True)

        index2 = [self.col2['device']]
        self.data2.set_index(index2, drop=True, inplace=True)

    def sti_place_num_index(self, segment: pd.DataFrame, col_segment: dict, into: str,
                            front_time: int, back_time: int, space_distance: int,
                            place_top=None, num_top=None):
        self.into = into
        if self.into == "device":
            self.space_distance = space_distance
            self.device_distance = self._create_device_distance_table()
            need_col1 = [self.col1['user'], self.col1['time'], self.col1['device']]
            self.data1 = self.data1[need_col1]
            need_col2 = [self.col2['user'], self.col2['time'], self.col2['device']]
            self.data2 = self.data2[need_col2]
            self.front_time = front_time
            self.back_time = back_time
            self.place_top = place_top
            self.num_top = num_top

            self._create_index()
            segment.reset_index(drop=True, inplace=True)
            segment[self.col2['user']] = segment.parallel_apply(
                lambda row: self._candidate_user_index(
                    row[col_segment['user1']], row[col_segment['start_time']], row[col_segment['end_time']]
                    ), axis=1)
            pairs = segment.dropna()
            pairs_col = {"user1": col_segment['user1'],
                         "user2": self.col2['user'],  # [user2, user2, ...] all candidate
                         "segment": col_segment['segment'],
                         "start_time": col_segment['start_time'],
                         "end_time": col_segment['end_time']}
            pairs = pairs[list(pairs_col.values())]
            return pairs, pairs_col
