import pandas as pd
from geopy.distance import geodesic
from pandarallel import pandarallel
pandarallel.initialize()

from libtrajectory.utils.coordinate import device_distance


class FeatureExtraction(object):
    def __init__(self, data1=None, col1=None, data2=None, col2=None):
        """
        :param data1: pd.DataFrame
            trajectory data, data1在data2内寻找关系对的
        :param col1: list
            The columns names corresponding to trajectory data1,
            {user: data1 column name,
            time: data1 column name,
            longitude: data1 column name,
            latitude: data1 column name,
            device: data1 column name, ...}
        :param data2: pd.DataFrame
            trajectory data
        :param col2: list
            The columns names corresponding to trajectory data2,
            {user: data2 column name,
            time: data2 column name,
             ...}
        """
        self.data1 = data1
        self.col1 = col1
        self.data2 = data2
        self.col2 = col2
        self.device_distance = None
        self.front_time = None
        self.back_time = None
        self.space_distance = None
        self.deduplication_seq1 = None
        self.deduplication_seq2 = None
        self.cache = [0, 0, 0]
        self.cache_sequence1 = None
        self.cache_feature1 = None

    def _create_device_distance_table(self, index=False):
        if index:
            device1_col = [self.col1['device'], self.col1['longitude'], self.col1['latitude']]
            device1 = self.data1.reset_index(level=self.col1['device'])
            device1 = device1[device1_col]
            device1 = device1.drop_duplicates(subset=device1_col)

            device2_col = [self.col2['device'], self.col2['longitude'], self.col2['latitude']]
            device2 = self.data2.reset_index(level=self.col2['device'])
            device2 = device2[device2_col]
            device2 = device2.drop_duplicates(subset=device2_col)

            distance_name = "distance"  # distance between devices
            df = device_distance(
                device1, device1_col, device2, device2_col, self.space_distance[-1], distance_name
            )
        else:
            device1_col = [self.col1['device'], self.col1['longitude'], self.col1['latitude']]
            device1 = self.data1[device1_col]
            device1 = device1.drop_duplicates(subset=device1_col)
            device2_col = [self.col2['device'], self.col2['longitude'], self.col2['latitude']]
            device2 = self.data2[device2_col]
            device2 = device2.drop_duplicates(subset=device2_col)
            distance_name = "distance"  # distance between devices
            df = device_distance(
                device1, device1_col, device2, device2_col, self.space_distance[-1], distance_name
            )
        return df

    def _feature1(self, sequence1):
        """
        :param sequence1:
           length1: sequence1 length, 代表用户有多少的轨迹点
           device1_num: sequence1 device number, 代表用户被多少个设备采集到
           time1_diff: sequence1首末次时间差
           count_distance1: 总移动空间距离（按时间线计算相邻device的空间距离，并对其求和）
           max_distance1: 最大移动空间距离（计算sequence1内device之间距离，取最大值）
        :return:
        """
        length1 = sequence1.shape[0]
        device1_num = sequence1[self.col1['device']].unique().__len__()
        time1_diff = sequence1[self.col1['time']].max() - sequence1[self.col1['time']].min()

        device_list = sequence1[self.col1['device']].tolist()
        longitude = sequence1[self.col1['latitude']].tolist()
        latitude = sequence1[self.col1['latitude']].tolist()
        if device_list.__len__() == 1:
            count_distance1 = 0
        else:
            # [user, time, longitude, latitude, device / None, poi / None]
            distance = [geodesic((latitude[i - 1], longitude[i - 1]), (latitude[i], longitude[i])).m
                        for i in range(device_list.__len__()) if i]
            count_distance1 = sum(distance)

        device_unique = list(set(device_list))
        if device_unique.__len__() == 1:
            max_distance1 = 0
        else:
            distance = []
            for i, d in enumerate(device_unique):
                index1 = device_list.index(d)
                for j in range(i + 1, device_unique.__len__()):
                    index2 = device_list.index(device_unique[j])
                    distance.append(geodesic((latitude[index1], longitude[index1]),
                                             (latitude[index2], longitude[index2])).m)
            max_distance1 = max(distance) if distance.__len__() > 1 else distance[0]

        feature1 = {
            "length1": length1,
            "device1_num": device1_num,
            "time1_diff": time1_diff,
            "count_distance1": count_distance1,
            "max_distance1": max_distance1
        }

        return feature1

    def _feature2(self, sequence2):
        """
        :param sequence2:
           length2: sequence1 length, 代表用户有多少的轨迹点
           device2_num: sequence2 device number, 代表用户被多少个设备采集到
           count_distance2: 总移动空间距离（按时间线计算相邻device的空间距离，并对其求和）
           max_distance2: 最大移动空间距离（计算sequence2内device之间距离，取最大值）
        :return:
        """
        length2 = sequence2.shape[0]
        device2_num = sequence2[self.col2['device']].unique().__len__()

        device_list = sequence2[self.col2['device']].tolist()
        longitude = sequence2[self.col2['latitude']].tolist()
        latitude = sequence2[self.col2['latitude']].tolist()
        if device_list.__len__() == 1:
            count_distance2 = 0
        else:
            # [user, time, longitude, latitude, device / None, poi / None]
            distance = [geodesic((latitude[i - 1], longitude[i - 1]), (latitude[i], longitude[i])).m
                        for i in range(device_list.__len__()) if i]
            count_distance2 = sum(distance)

        device_unique = list(set(device_list))
        if device_unique.__len__() == 1:
            max_distance2 = 0
        else:
            distance = []
            for i, d in enumerate(device_unique):
                index1 = device_list.index(d)
                for j in range(i + 1, device_unique.__len__()):
                    index2 = device_list.index(device_unique[j])
                    distance.append(geodesic((latitude[index1], longitude[index1]),
                                             (latitude[index2], longitude[index2])).m)
            max_distance2 = max(distance) if distance.__len__() > 1 else distance[0]

        feature2 = {
            "length2": length2,
            "device2_num": device2_num,
            "count_distance2": count_distance2,
            "max_distance2": max_distance2
        }

        return feature2

    def _feature3(self, sequence3):
        """
        :param sequence3:
            sti_length: sequence3 length, 代表user1与user2产生碰撞的轨迹点数量
            sti_user1_num: sequence3中user1的轨迹点去重后的数量
            sti_user2_num: sequence3中user2的轨迹点去重后的数量
            sti_device1_num: sequence3中user1的设备数量
            sti_device2_num: sequence3中user2的设备数量
            sti_time1_diff: 首末次时间差
            sti_count_distance1: 总移动空间距离（按时间线计算相邻device1的空间距离，并对其求和）
            sti_max_distance1: 最大移动空间距离（计算sequence3内device1之间距离，取最大值）
        :return:
        """
        feature3 = {}
        for threshold in self.space_distance:
            sti_col = [self.col2['user'] + str(threshold),
                       self.col2['time'] + str(threshold),
                       self.col2['device'] + str(threshold)]
            col = list(self.col1.values()) + sti_col
            seq_th: pd.DataFrame = sequence3[col]  # sequence3 space threshold
            seq_th = seq_th.dropna(subset=sti_col)
            if not seq_th.shape[0]:
                dic = {f"sti_length_{str(threshold)}": 0,
                       f"sti_user1_num_{str(threshold)}": 0,
                       f"sti_user2_num_{str(threshold)}": 0,
                       f"sti_device1_num_{str(threshold)}": 0,
                       f"sti_device2_num_{str(threshold)}": 0,
                       f"sti_time1_diff_{str(threshold)}": 0,
                       f"sti_count_distance1_{str(threshold)}": 0,
                       f"sti_max_distance1_{str(threshold)}": 0
                       }
                feature3 = {**feature3, **dic}
                continue
            else:
                sti_length = seq_th.shape[0]
                sti_user1_num = seq_th.drop_duplicates([self.col1['device'], self.col1['time']]).shape[0]
                sti_user2_num = seq_th.drop_duplicates(
                    [self.col2['device'] + str(threshold), self.col2['time'] + str(threshold)]).shape[0]
                sti_device1_num = seq_th[self.col1['device']].unique().__len__()
                sti_device2_num = seq_th[self.col2['device'] + str(threshold)].unique().__len__()

                if seq_th.shape[0] == 1:
                    sti_time1_diff = 0
                else:
                    sti_time1_diff = seq_th[self.col1['time']].max() - seq_th[self.col1['time']].min()

                device1 = seq_th[self.col1['device']].tolist()
                longitude = seq_th[self.col1['latitude']].tolist()
                latitude = seq_th[self.col1['latitude']].tolist()
                if device1.__len__() == 1:
                    sti_count_distance1 = 0
                else:
                    distance = [geodesic((latitude[i - 1], longitude[i - 1]), (latitude[i], longitude[i])).m
                                for i in range(device1.__len__()) if i]
                    sti_count_distance1 = sum(distance)

                device_unique = list(set(device1))
                if device_unique.__len__() == 1:
                    sti_max_distance1 = 0
                else:
                    distance = []
                    for i, d in enumerate(device_unique):
                        index1 = device1.index(d)
                        for j in range(i + 1, device_unique.__len__()):
                            index2 = device1.index(device_unique[j])
                            distance.append(geodesic((latitude[index1], longitude[index1]),
                                                     (latitude[index2], longitude[index2])).m)
                    sti_max_distance1 = max(distance) if distance.__len__() > 1 else distance[0]

                dic = {f"sti_length_{str(threshold)}": sti_length,
                       f"sti_user1_num_{str(threshold)}": sti_user1_num,
                       f"sti_user2_num_{str(threshold)}": sti_user2_num,
                       f"sti_device1_num_{str(threshold)}": sti_device1_num,
                       f"sti_device2_num_{str(threshold)}": sti_device2_num,
                       f"sti_time1_diff_{str(threshold)}": sti_time1_diff,
                       f"sti_count_distance1_{str(threshold)}": sti_count_distance1,
                       f"sti_max_distance1_{str(threshold)}": sti_max_distance1
                       }
                feature3 = {**feature3, **dic}

        return feature3

    def _device_time_sti(self, device1, time1, sequence2):
        sti_start_time = time1 - self.back_time
        sti_end_time = time1 + self.front_time
        device2 = self.device_distance[
            self.device_distance[self.col1['device']].isin([device1])][self.col2['device']].unique().tolist()
        if not device2:
            return None, None, None
        sti = sequence2[(sequence2[self.col2['device']].isin(device2)) & (
            sequence2[self.col2['time']].isin(range(sti_start_time, sti_end_time + 1)))]
        if not sti.shape[0]:
            return None, None, None
        sti = sti.to_dict(orient="list")
        return sti[self.col2['user']], sti[self.col2['time']], sti[self.col2['device']]

    def _extraction(self, user1, user2, start_time, end_time):
        # sequence1: data1 user1 trajectory
        if self.cache[0] == user1 and self.cache[1] == start_time and self.cache[2] == end_time:
            sequence1 = self.cache_sequence1
            feature1 = self.cache_feature1
        else:
            sequence1: pd.DataFrame = self.data1[(self.data1[self.col1['user']].isin([user1])) &
                                                 (self.data1[self.col1['time']].isin(range(start_time, end_time + 1)))]
            sequence1.sort_values(self.col1['time'], ascending=True, inplace=True)
            self.sequence1 = sequence1
            # Todo sequence1 deduplication
            feature1 = self._feature1(sequence1)

            self.cache = [user1, start_time, end_time]
            self.cache_sequence1 = sequence1
            self.cache_feature1 = feature1

        # sequence2: data2 user2 trajectory
        sti_start_time = start_time - self.back_time
        sti_end_time = end_time + self.front_time
        sequence2: pd.DataFrame = self.data2[(self.data2[self.col2['user']].isin([user2])) & (
            self.data2[self.col2['time']].isin(range(sti_start_time, sti_end_time + 1)))]
        sequence2.sort_values(self.col2['time'], ascending=True, inplace=True)
        feature2 = self._feature2(sequence2)
        # Todo sequence deduplication

        # sequence3: spatiotemporal intersection sequence
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        pd.set_option('display.max_columns', None)
        sequence3: pd.DataFrame = sequence1.copy()
        sti_lis_max = [c + str(self.space_distance[-1]) for c in
                       [self.col2['user'], self.col2['time'], self.col2['device']]]
        sequence3[sti_lis_max] = sequence3.apply(
            lambda row: self._device_time_sti(row[self.col1['device']], row[self.col1['time']], sequence2)
            , result_type="expand", axis=1)
        sequence3 = sequence3.dropna()
        sequence3 = sequence3.reset_index(drop=True)
        sequence3 = sequence3.apply(pd.Series.explode)

        if self.space_distance.__len__() > 1:  # space_distance have many threshold
            sequence3['distance'] = sequence3.apply(
                lambda row: self.device_distance[
                    (self.device_distance[self.col1['device']].isin([row[self.col1['device']]])) &
                    (self.device_distance[self.col2['device']].isin(
                        [row[self.col2['device'] + str(self.space_distance[-1])]]))
                    ].distance.values[0], axis=1)
            pd.set_option('display.max_columns', None)

            for threshold in self.space_distance[:-1]:
                for c in [self.col2['user'], self.col2['time'], self.col2['device']]:
                    sequence3[c + str(threshold)] = sequence3.apply(
                        lambda row: row[c + str(self.space_distance[-1])] if row['distance'] <= threshold else None,
                        axis=1)
        feature3 = self._feature3(sequence3)

        feature = {**feature1, **feature2, **feature3}
        return feature

    def sequence(self, pairs: pd.DataFrame, col_pairs: dict,
                 front_time: int, back_time: int, space_distance: list,
                 deduplication_seq1=None, deduplication_seq2=None):
        """
        通过sequence来提取特征
        sequence1: user1 trajectory sequence,
            if type(deduplication_seq1) is int, sequence1 time deduplication
        sequence2: user2 trajectory sequence
            if type(deduplication_seq2) is int, sequence1 time deduplication
        sequence3: spatiotemporal intersection sequence,
            Trajectory points where user1 trajectory sequence and user2 trajectory sequence generate spatiotemporal intersection
        :param pairs: pd.DataFrame
            需要提取特征的候选关系对
        :param col_pairs: dict
            The columns names corresponding to trajectory pair,
            {user1: pairs column name,
            user2: pairs column name,
            segment: pairs column name,
            start_time: pairs column name,
            end_time: pairs column name, ...}
        :param front_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param back_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param space_distance: list, unit is meter
            [空间阈值1, 空间阈值2, ...] 由小到大
            时空交集的空间阈值, data1的轨迹点地理位置为p，在进行空间交集时，寻找data2在[p-space, p+space]距离内的轨迹点。
        :param deduplication_seq1: None or int, unit second
            None: sequence1 does not undergo time deduplication
            int: sequence1 time deduplication  Todo
        :param deduplication_seq2: None or int, unit second
            None: sequence2 does not undergo time deduplication
            int: sequence2 time deduplication  Todo
        :return: pd.DataFrame,  columns is [user1, user2, segment, start_time, end_time, feature1, feature2, ...]
        """
        self.front_time = front_time
        self.back_time = back_time
        self.space_distance = space_distance
        self.deduplication_seq1 = deduplication_seq1
        self.deduplication_seq2 = deduplication_seq2

        if self.col1['device'] is not None and self.col1.get("poi", None) is None:
            self.device_distance = self._create_device_distance_table()

            # col_pair = [user1, user2, pair, start_time, end_time]
            pairs["feature"] = pairs.parallel_apply(lambda row: self._extraction(
                row[col_pairs[0]], row[col_pairs[1]], row[col_pairs[3]], row[col_pairs[4]]), axis=1)
            pairs.reset_index(drop=True, inplace=True)
            feature = pd.concat([pairs, pairs['feature'].apply(pd.Series)], axis=1).drop('feature', axis=1)

            return feature, feature.columns

    def _feature1_index(self, sequence1: pd.DataFrame):
        """
        :param sequence1:
           length1: sequence1 length, 代表用户有多少的轨迹点
           device1_num: sequence1 device number, 代表用户被多少个设备采集到
           time1_diff: sequence1首末次时间差
           count_distance1: 总移动空间距离（按时间线计算相邻device的空间距离，并对其求和）
           max_distance1: 最大移动空间距离（计算sequence1内device之间距离，取最大值）
        :return:
        """
        time1_values = sequence1.index.get_level_values(self.col1['time'])
        device1_values = sequence1.index.get_level_values(self.col1['device'])

        length1 = sequence1.shape[0]
        device1_num = device1_values.unique().__len__()
        time1_diff = time1_values.max() - time1_values.min()

        device1_list = device1_values.tolist()
        longitude = sequence1[self.col1['latitude']].tolist()
        latitude = sequence1[self.col1['latitude']].tolist()
        if device1_list.__len__() == 1:
            count_distance1 = 0
        else:
            # [user, time, longitude, latitude, device / None, poi / None]
            distance = [geodesic((latitude[i - 1], longitude[i - 1]), (latitude[i], longitude[i])).m
                        for i in range(device1_list.__len__()) if i]
            count_distance1 = sum(distance)

        device_unique = list(set(device1_list))
        if device_unique.__len__() == 1:
            max_distance1 = 0
        else:
            distance = []
            for i, d in enumerate(device_unique):
                index1 = device1_list.index(d)
                for j in range(i + 1, device_unique.__len__()):
                    index2 = device1_list.index(device_unique[j])
                    distance.append(geodesic((latitude[index1], longitude[index1]),
                                             (latitude[index2], longitude[index2])).m)
            max_distance1 = max(distance) if distance.__len__() > 1 else distance[0]

        feature1 = {
            "length1": length1,
            "device1_num": device1_num,
            "time1_diff": time1_diff,
            "count_distance1": count_distance1,
            "max_distance1": max_distance1
        }

        return feature1

    def _feature2_index(self, sequence2: pd.DataFrame):
        """
        :param sequence2:
           length2: sequence1 length, 代表用户有多少的轨迹点
           device2_num: sequence2 device number, 代表用户被多少个设备采集到
           count_distance2: 总移动空间距离（按时间线计算相邻device的空间距离，并对其求和）
           max_distance2: 最大移动空间距离（计算sequence2内device之间距离，取最大值）
        :return:
        """
        device2_values = sequence2.index.get_level_values(self.col2['device'])

        length2 = sequence2.shape[0]
        device2_num = device2_values.unique().__len__()

        device_list = device2_values.tolist()
        longitude = sequence2[self.col2['latitude']].tolist()
        latitude = sequence2[self.col2['latitude']].tolist()
        if device_list.__len__() == 1:
            count_distance2 = 0
        else:
            distance = [geodesic((latitude[i - 1], longitude[i - 1]), (latitude[i], longitude[i])).m
                        for i in range(device_list.__len__()) if i]
            count_distance2 = sum(distance)

        device_unique = list(set(device_list))
        if device_unique.__len__() == 1:
            max_distance2 = 0
        else:
            distance = []
            for i, d in enumerate(device_unique):
                index1 = device_list.index(d)
                for j in range(i + 1, device_unique.__len__()):
                    index2 = device_list.index(device_unique[j])
                    distance.append(geodesic((latitude[index1], longitude[index1]),
                                             (latitude[index2], longitude[index2])).m)
            max_distance2 = max(distance) if distance.__len__() > 1 else distance[0]

        feature2 = {
            "length2": length2,
            "device2_num": device2_num,
            "count_distance2": count_distance2,
            "max_distance2": max_distance2
        }

        return feature2

    def _extraction_index(self, user1, user2, start_time, end_time):
        # sequence1: data1 user1 trajectory
        if self.cache[0] == user1 and self.cache[1] == start_time and self.cache[2] == end_time:
            sequence1 = self.cache_sequence1
            feature1 = self.cache_feature1
        else:
            sequence1 = self.data1.query(
                f"({self.col1['user']} == '{user1}') & "
                f"({end_time} >= {self.col1['time']}) & ({self.col1['time']} >= {start_time})"
            )
            self.sequence1 = sequence1
            # Todo sequence1 deduplication
            feature1 = self._feature1_index(sequence1)

            self.cache = [user1, start_time, end_time]
            self.cache_sequence1 = sequence1
            self.cache_feature1 = feature1

        # sequence2: data2 user2 trajectory
        sti_start_time = start_time - self.back_time
        sti_end_time = end_time + self.front_time
        sequence2 = self.data2.query(
            f"({self.col2['user']} == '{user2}') & "
            f"({sti_end_time}) >= {self.col2['time']} & ({self.col2['time']} >= {sti_start_time})"
        )
        feature2 = self._feature2_index(sequence2)
        # Todo sequence deduplication

        # sequence3: spatiotemporal intersection sequence
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        pd.set_option('display.max_columns', None)
        sequence3: pd.DataFrame = sequence1.copy()
        sti_lis_max = [c + str(self.space_distance[-1]) for c in
                       [self.col2['user'], self.col2['time'], self.col2['device']]]
        sequence3[sti_lis_max] = sequence3.apply(
            lambda row: self._device_time_sti(row[self.col1['device']], row[self.col1['time']], sequence2)
            , result_type="expand", axis=1)
        sequence3 = sequence3.dropna()
        sequence3 = sequence3.reset_index(drop=True)
        sequence3 = sequence3.apply(pd.Series.explode)

        if self.space_distance.__len__() > 1:  # space_distance have many threshold
            sequence3['distance'] = sequence3.apply(
                lambda row: self.device_distance[
                    (self.device_distance[self.col1['device']].isin([row[self.col1['device']]])) &
                    (self.device_distance[self.col2['device']].isin(
                        [row[self.col2['device'] + str(self.space_distance[-1])]]))
                    ].distance.values[0], axis=1)
            pd.set_option('display.max_columns', None)

            for threshold in self.space_distance[:-1]:
                for c in [self.col2['user'], self.col2['time'], self.col2['device']]:
                    sequence3[c + str(threshold)] = sequence3.apply(
                        lambda row: row[c + str(self.space_distance[-1])] if row['distance'] <= threshold else None,
                        axis=1)
        feature3 = self._feature3(sequence3)

        feature = {**feature1, **feature2, **feature3}
        return feature

    def sequence_index(self, pairs: pd.DataFrame, col_pairs: dict,
                       front_time: int, back_time: int, space_distance: list,
                       deduplication_seq1=None, deduplication_seq2=None):
        """
        通过sequence来提取特征
        sequence1: user1 trajectory sequence,
            if type(deduplication_seq1) is int, sequence1 time deduplication
        sequence2: user2 trajectory sequence
            if type(deduplication_seq2) is int, sequence1 time deduplication
        sequence3: spatiotemporal intersection sequence,
            Trajectory points where user1 trajectory sequence and user2 trajectory sequence generate spatiotemporal intersection
        :param pairs: pd.DataFrame
            需要提取特征的候选关系对
        :param col_pairs: dict
            The columns names corresponding to trajectory pair,
            {user1: pairs column name,
            user2: pairs column name,
            segment: pairs column name,
            start_time: pairs column name,
            end_time: pairs column name, ...}
        :param front_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param back_time: int, unit is second
            时空交集的时间阈值, data1的轨迹点时间为t，在进行时间交集时，寻找data2在[t-back_time, t+front_time]时间内的轨迹点。
        :param space_distance: list, unit is meter
            [空间阈值1, 空间阈值2, ...] 由小到大
            时空交集的空间阈值, data1的轨迹点地理位置为p，在进行空间交集时，寻找data2在[p-space, p+space]距离内的轨迹点。
        :param deduplication_seq1: None or int, unit second
            None: sequence1 does not undergo time deduplication
            int: sequence1 time deduplication  Todo
        :param deduplication_seq2: None or int, unit second
            None: sequence2 does not undergo time deduplication
            int: sequence2 time deduplication  Todo
        :return: pd.DataFrame,  columns is [user1, user2, segment, start_time, end_time, feature1, feature2, ...]
        """
        self.front_time = front_time
        self.back_time = back_time
        self.space_distance = space_distance
        self.deduplication_seq1 = deduplication_seq1
        self.deduplication_seq2 = deduplication_seq2

        if self.col1['device'] is not None and self.col1.get("poi", None) is None:
            self.device_distance = self._create_device_distance_table()

            # col_pair = [user1, user2, pair, start_time, end_time]
            pairs["feature"] = pairs.apply(lambda row: self._extraction_index(
                row[col_pairs[0]], row[col_pairs[1]], row[col_pairs[3]], row[col_pairs[4]]), axis=1)
            pairs.reset_index(drop=True, inplace=True)
            feature = pd.concat([pairs, pairs['feature'].apply(pd.Series)], axis=1).drop('feature', axis=1)

            col = feature.columns
            feature_col = col_pairs.copy()
            feature_col['feature'] = [c for c in col if c not in col_pairs.values()]
            return feature, feature_col
