import logging
import io
import math
import time

import pandas as pd
from math import cos, radians


class Preprocessor(object):
    def __init__(self, data_path, test_path):
        self.data_path = data_path
        self.test_path = test_path
        self.data1 = None  # Active
        self.data2 = None  # Passive
        logging.info(f"self.path={self.data_path}")

    def run(self):
        self.traj_loader()
        self.cleaner()
        self.space2coor()  # 空间建模
        self.time2coor()  # 时间建模
        self.space2vector()
        self.time2vector()
        self.st2vector()
        self.save()
        return self.train_data, self.test_data, self.st_vec, self.stid_counts

    def traj_loader(self):
        logging.info("data loading...")
        data = pd.read_csv(self.data_path, dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
        self.data1 = data[['tid', 'time', 'lat', 'lon', 'did']].copy()
        self.data2 = data[['tid', 'm_time', 'm_lat', 'm_lon', 'm_did']].copy()
        self.data2.rename(columns={'m_time': 'time', 'm_lat': 'lat', 'm_lon': 'lon', 'm_did': 'did'}, inplace=True)
        self.data2 = self.data2[self.data2['time'] != 0]

        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info: {buffer.getvalue()}")
        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info: {buffer.getvalue()}")

        self.test = {}
        for k, v in self.test_path.items():
            data = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
            self.test[k] = data

        logging.info("data load completed")

    def cleaner(self):
        logging.info("data clean...")
        self.data1.dropna(inplace=True)
        self.data1.drop_duplicates(inplace=True)
        self.data1.sort_values(['time'], inplace=True)
        self.data2.dropna(inplace=True)
        self.data2.drop_duplicates(inplace=True)
        self.data2.sort_values(['time'], inplace=True)
        logging.info("data clean completed")

    def space2coor(self):
        logging.info("space coordinate...")
        lat0 = min([self.data1.lat.min(), self.data2.lat.min()])
        lat1 = max([self.data1.lat.max(), self.data2.lat.max()])
        lon0 = min([self.data1.lon.min(), self.data2.lon.min()])
        lon1 = max([self.data1.lon.max(), self.data2.lon.max()])
        logging.info(f"lat0: {lat0}, lat1: {lat1}, lon0: {lon0}, lon1: {lon1}")

        # 获取区域的lat、lon步长
        deci = 5  # decimal 小数点后的精度
        distance = 10 * 1000  # m
        r = 6371393  # 地球半径 单位m
        lat_step = (distance / (r * cos(radians(0)))) * (180 / 3.1415926)
        lon_step = (distance / r) * (180 / 3.1415926)
        lat_len, lon_len = abs(lat0) + abs(lat1), abs(lon0) + abs(lon1)
        lat_size, lon_size = math.ceil(lat_len / lat_step), math.ceil(lon_len / lon_step)
        logging.info(f"lat_len: {lat_len}, lon_len： {lon_len}， lat_step: {lat_step}, lon_step: {lon_step}")

        # 生成区域节点的lat和lon值
        lat_lis, lon_lis = [lat0], [lon0]
        for _ in range(lat_size):
            lat_lis.append(round(lat_lis[-1] + lat_step, deci))
        for _ in range(lon_size):
            lon_lis.append(round(lon_lis[-1] + lon_step, deci))
        lat_lis[-1], lon_lis[-1] = lat1, lon1

        logging.info("data1 space coordinate...")
        space_num = SpaceNum(lat_lis, lon_lis)
        test = self.data1.copy()
        self.data1 = space_num.get(test)
        logging.info("data1 space coordinate completed")
        logging.info("data2 space coordinate...")
        test = self.data2.copy()
        self.data2 = space_num.get(test)
        logging.info("data2 space coordinate completed")

    def time2coor(self, groupby='month'):
        logging.info("time coordinate...")
        interval = 10 * 60
        logging.info(f"time interval={interval}s")
        if groupby == 'month':
            self.data1['tgroup'] = self.data1['time'].map(lambda t: time.localtime(t).tm_mon)
            self.data2['tgroup'] = self.data2['time'].map(lambda t: time.localtime(t).tm_mon)
            logging.info("time coordinate group by month")
        if groupby == 'week':
            self.data1['tgroup'] = self.data1['time'].map(lambda t: time.localtime(t).tm_wday + 1)
            self.data2['tgroup'] = self.data2['time'].map(lambda t: time.localtime(t).tm_wday + 1)
            logging.info("time coordinate group by week")

        df1 = self.data1[['time', 'tgroup']].copy()
        df2 = self.data2[['time', 'tgroup']].copy()
        df = pd.concat([df1, df2])
        group = df.groupby('tgroup')['time'].agg(['min', 'max'])
        time_dict = {}

        for flag, v in group.iterrows():
            t0, t1 = int(v['min']), int(v['max'])
            size = math.ceil((t1 - t0) / interval)
            lis = [t0]
            for _ in range(size):
                lis.append(lis[-1] + interval)
            lis[-1] = t1
            time_dict[flag] = lis

        logging.info("data1 time coordinate...")
        timeid = TimeNum(time_dict)
        test = self.data1.copy()
        self.data1 = timeid.get(test)
        logging.info("data1 time coordinate completed")
        logging.info("data2 time coordinate...")
        test = self.data2.copy()
        self.data2 = timeid.get(test)
        logging.info("data2 time coordinate completed")

    def space2vector(self):
        logging.info("space2vector...")

        self.data1['latid'] = self.data1['spaceid'].map(lambda s: int(s.split('-')[0]))
        self.data2['latid'] = self.data2['spaceid'].map(lambda s: int(s.split('-')[0]))
        lat_max = int(max([self.data1.latid.max(), self.data2.latid.max()]))
        lat_len = (lat_max - 1).bit_length() + 4
        logging.info(f"lat_max={lat_max}, lat_len={lat_len}")

        self.data1['lonid'] = self.data1['spaceid'].map(lambda s: int(s.split('-')[1]))
        self.data2['lonid'] = self.data2['spaceid'].map(lambda s: int(s.split('-')[1]))
        lon_max = int(max([self.data1.lonid.max(), self.data2.lonid.max()]))
        lon_len = (lon_max - 1).bit_length() + 4
        logging.info(f"lon_max={lon_max}, lon_len={lon_len}")

        df1 = self.data1[['spaceid', 'latid', 'lonid']].copy()
        df2 = self.data2[['spaceid', 'latid', 'lonid']].copy()
        s_vec = pd.concat([df1, df2])
        s_vec.drop_duplicates(inplace=True)
        s_vec['vec'] = s_vec.apply(lambda row: self._2binary(row.latid, lat_len, row.lonid, lon_len), axis=1)
        self.s_vec = s_vec
        logging.info("space2vector completed")

    def time2vector(self):
        logging.info("time2vector...")
        self.data1['timeid0'] = self.data1['timeid'].map(lambda t: int(t.split("-")[0]))
        self.data2['timeid0'] = self.data2['timeid'].map(lambda t: int(t.split("-")[0]))
        timeid0_max = int(max([self.data1.timeid0.max(), self.data2.timeid0.max()]))
        timeid0_len = (timeid0_max - 1).bit_length() + 4
        logging.info(f"time group max={timeid0_max}, len={timeid0_len}")

        self.data1['timeid1'] = self.data1['timeid'].map(lambda t: int(t.split("-")[1]))
        self.data2['timeid1'] = self.data2['timeid'].map(lambda t: int(t.split("-")[1]))
        timeid1_max = int(max([self.data1.timeid1.max(), self.data2.timeid1.max()]))
        timeid1_len = (timeid1_max - 1).bit_length() + 4
        logging.info(f"max value within the time group={timeid1_max}, len={timeid1_len}")

        df1 = self.data1[['timeid', 'timeid0', 'timeid1']].copy()
        df2 = self.data2[['timeid', 'timeid0', 'timeid1']].copy()
        t_vec = pd.concat([df1, df2])
        t_vec.drop_duplicates(inplace=True)
        t_vec['vec'] = t_vec.apply(lambda row: self._2binary(row.timeid0, timeid0_len, row.timeid1, timeid1_len),
                                   axis=1)
        self.t_vec = t_vec
        logging.info("time2vector completed")

    def st2vector(self):
        logging.info("spatiotemporal2vector...")
        self.data1['stid'] = self.data1.apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)
        self.data2['stid'] = self.data2.apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)

        df1 = self.data1[['stid']].copy()
        df2 = self.data2[['stid']].copy()
        stid = pd.concat([df1, df2])
        stid.to_csv(f"{self.data_path[: -4]}_stid.csv", index=False)
        self.stid_counts = stid.stid.value_counts().to_dict()

        df1 = self.data1[['stid', 'spaceid', 'timeid']].copy()
        df2 = self.data2[['stid', 'spaceid', 'timeid']].copy()
        st_vec = pd.concat([df1, df2])
        st_vec.drop_duplicates(inplace=True)

        st_vec = st_vec.merge(self.s_vec, how='left')
        st_vec.rename(columns={"vec": "s_vec"}, inplace=True)
        st_vec = st_vec.merge(self.t_vec, how='left')
        st_vec.rename(columns={"vec": "t_vec"}, inplace=True)
        st_vec['vec'] = st_vec.apply(lambda x: x.s_vec[0] + x.s_vec[1] + x.t_vec[0] + x.t_vec[1], axis=1)
        st_vec = st_vec[['stid', 'spaceid', 'timeid', 'vec']]
        self.st_vec = st_vec
        logging.info("spatiotemporal2vector completed")

    def _2binary(self, num1, len1, num2=None, len2=None):
        bin_str1 = bin(num1)[2:]
        if len(bin_str1) < len1:
            padding1 = '0' * (len1 - len(bin_str1))
            bin_str1 = padding1 + bin_str1
        bin_vec1 = [int(i) for i in bin_str1]

        if num2 is None:
            return bin_vec1

        bin_str2 = bin(num2)[2:]
        if len(bin_str2) < len2:
            padding2 = '0' * (len2 - len(bin_str2))
            bin_str2 = padding2 + bin_str2
        bin_vec2 = [int(i) for i in bin_str2]

        bin_vec = [bin_vec1, bin_vec2]
        return bin_vec

    def _stid(self):
        df1 = self.data1[['stid']].copy()
        df2 = self.data2[['stid']].copy()
        self.stid = pd.concat([df1, df2])
        self.stid_counts = self.stid.stid.value_counts().to_dict()
        logging.info("stid, stid_counts completed")

    def save(self):
        logging.info("data save...")
        self._stid()
        self.s_vec.to_csv(f"{self.data_path[: -4]}_s_vec.csv", index=False)
        self.t_vec.to_csv(f"{self.data_path[: -4]}_t_vec.csv", index=False)
        self.st_vec.to_csv(f"{self.data_path[: -4]}_st_vec.csv", index=False)
        self.stid.to_csv(f"{self.data_path[: -4]}_stid.csv", index=False)

        columns = ['tid', 'time', 'lat', 'lon', 'spaceid', 'timeid', 'stid']
        data1 = self.data1[columns]
        data1.to_csv(f"{self.data_path[: -4]}_data1.csv", index=False)
        data2 = self.data2[columns]
        data2.to_csv(f"{self.data_path[: -4]}_data2.csv", index=False)

        test_tid = []
        self.test_data = {}
        for k, df in self.test.items():
            tid = df.tid.unique().tolist()
            test_tid += tid
            df1 = data1.query(f"tid in {tid}").copy()
            df1.reset_index(drop=True, inplace=True)
            df1.to_csv(f"{self.data_path[: -4]}_{k}_data1.csv", index=False)
            df2 = data2.query(f"tid in {tid}").copy()
            df2.reset_index(drop=True, inplace=True)
            df2.to_csv(f"{self.data_path[: -4]}_{k}_data2.csv", index=False)
            self.test_data[k] = [df1, df2]

        data1 = data1.query(f"tid not in {test_tid}").copy()
        data1.reset_index(drop=True, inplace=True)
        data1.to_csv(f"{self.data_path[: -4]}_train_data1.csv", index=False)
        data2 = data2.query(f"tid not in {test_tid}").copy()
        data2.reset_index(drop=True, inplace=True)
        data2.to_csv(f"{self.data_path[: -4]}_train_data2.csv", index=False)
        self.train_data = [data1, data2]
        logging.info("data save completed")

    def load(self):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=12)
        # logging.info("spatiotemporal vector loading...")
        # st_vec = pd.read_csv(f"{self.data_path[: -4]}_st_vec.csv")
        # st_vec['vec'] = st_vec['vec'].parallel_map(lambda x: eval(x))
        # logging.info("spatiotemporal vector load completed")

        logging.info("train data loading...")
        dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'spaceid': str, 'timeid': str, 'stid': str}
        train_data1 = pd.read_csv(f"{self.data_path[: -4]}_train_data1.csv", dtype=dic)
        train_data2 = pd.read_csv(f"{self.data_path[: -4]}_train_data2.csv", dtype=dic)
        train_data = [train_data1, train_data2]
        logging.info("train data load completed")

        logging.info("test data loading...")
        test_data = {}
        for k, v in self.test_path.items():
            df1 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data1.csv", dtype=dic)
            df2 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data2.csv", dtype=dic)
            test_data[k] = [df1, df2]
        logging.info(f"test data load completed")

        logging.info("trajectory points and spatiotemporal_id loading...")
        stid_counts = pd.read_csv(f"{self.data_path[: -4]}_stid.csv").stid.value_counts().to_dict()
        logging.info("trajectory points and spatiotemporal_id load completed")

        return train_data, test_data, stid_counts
        # return train_data, test_data, st_vec, stid_counts

    def get(self, method="load"):
        if method == "load":
            return self.load()
        if method == "run":
            return self.run()


class SpaceNum(object):
    def __init__(self, lat_lis, lon_lis):
        self.lat_lis = lat_lis
        self.lon_lis = lon_lis

    def _create_spaceid(self, lat, lon):
        latf = None
        lat0 = self.lat_lis[0]
        for i, lat1 in enumerate(self.lat_lis[1:]):
            if lat0 <= lat <= lat1:
                latf = i + 1
                break
            lat0 = lat1
        if latf is None:
            logging.critical(f"lat={lat} 找不到对应的空间切割编号")

        lonf = None
        lon0 = self.lon_lis[0]
        for j, lon1 in enumerate(self.lon_lis[1:]):
            if lon0 <= lon <= lon1:
                lonf = j + 1
                break
            lon0 = lon1
        if lonf is None:
            logging.critical(f"lon={lon} 找不到对应的空间切割编号")

        return f"{latf}-{lonf}"

    def get(self, df):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=4)
        df['spaceid'] = df.parallel_apply(lambda row: self._create_spaceid(row.lat, row.lon), axis=1)
        return df


class TimeNum(object):
    def __init__(self, time_dict):
        self.time_dict = time_dict

    def _create_timeid(self, tgroup, t):
        t_lis = self.time_dict[tgroup]
        t0 = t_lis[0]
        for i, t1 in enumerate(t_lis[1:]):
            if t0 <= t <= t1:
                return f"{tgroup}-{i + 1}"
            t0 = t1
        logging.critical(f"t={t} 找不到相应的时间分割编号")

    def get(self, df):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=4)
        df['timeid'] = df.apply(lambda row: self._create_timeid(row.tgroup, row.time), axis=1)
        return df
