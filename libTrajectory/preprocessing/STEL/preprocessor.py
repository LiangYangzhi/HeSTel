import logging
import io
import math
import time

import pandas as pd
from math import cos, radians


class Preprocessor(object):
    def __init__(self, data_path, test_path={}):
        self.data_path = data_path
        self.test_path = test_path
        self.data1 = None  # Active
        self.data2 = None  # Passive
        logging.info(f"self.path={self.data_path}")

    def run(self):
        self.loader()
        self.cleaner()
        self.space2coor()  # 空间建模
        self.time2coor()  # 时间建模
        self.space2vector()  # 空间编号转向量
        self.time2vector()  # 时间编号转向量
        self.st2vector()  # 时空编号转向量

    def loader(self):
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
        distance = 100 * 1000  # m
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
        self.lat_lis, self.lon_lis = lat_lis, lon_lis

        logging.info("data1 space coordinate...")
        self.data1['spaceid'] = self.data1.apply(lambda row: self._create_spaceid(row.lat, row.lon), axis=1)
        logging.info("data1 space coordinate completed")
        logging.info("data2 space coordinate...")
        self.data2['spaceid'] = self.data2.apply(lambda row: self._create_spaceid(row.lat, row.lon), axis=1)
        logging.info("data2 space coordinate completed")

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

        self.time_dict = time_dict
        logging.info("data1 time coordinate...")
        self.data1['timeid'] = self.data1.apply(lambda row: self._create_timeid(row.tgroup, row.time), axis=1)
        logging.info("data1 time coordinate completed")
        logging.info("data2 time coordinate...")
        self.data2['timeid'] = self.data2.apply(lambda row: self._create_timeid(row.tgroup, row.time), axis=1)
        logging.info("data2 time coordinate completed")

    def _create_timeid(self, tgroup, t):
        t_lis = self.time_dict[tgroup]
        t0 = t_lis[0]
        for i, t1 in enumerate(t_lis[1:]):
            if t0 <= t <= t1:
                return f"{tgroup}-{i + 1}"
            t0 = t1
        logging.critical(f"t={t} 找不到相应的时间分割编号")

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
        s_vec.to_csv(f"{self.data_path[: -4]}_space_vec.csv", index=False)
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
        t_vec['vec'] = t_vec.apply(lambda row: self._2binary(row.timeid0, timeid0_len, row.timeid1, timeid1_len), axis=1)
        t_vec.to_csv(f"{self.data_path[: -4]}_time_vec.csv", index=False)
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
        st_vec.to_csv(f"{self.data_path[: -4]}_st_vec.csv", index=False)
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

    def get(self):
        if self.data1 is None:
            logging.info("train data loading...")
            dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str, 'spaceid': str, 'timeid': str, 'stid': str}
            train_data1 = pd.read_csv(f"{self.data_path[: -4]}_train_data1.csv", dtype=dic)
            train_data2 = pd.read_csv(f"{self.data_path[: -4]}_train_data2.csv", dtype=dic)
            train_data = [train_data1, train_data2]
            logging.info("train data load completed")

            logging.info("test data loading...")
            test_data = {}
            for k, v in self.test_path.items():
                data1 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data1.csv", dtype=dic)
                data2 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data2.csv", dtype=dic)
                test_data[k] = [data1, data2]
            logging.info(f"test data load completed")

            logging.info("spatiotemporal vector loading...")
            st_vec = pd.read_csv(f"{self.data_path[: -4]}_st_vec.csv")
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=24)
            st_vec['s_vec'] = st_vec['s_vec'].parallel_map(lambda x: eval(x))
            st_vec['t_vec'] = st_vec['t_vec'].parallel_map(lambda x: eval(x))
            st_vec['vec'] = st_vec['vec'].parallel_map(lambda x: eval(x))
            self.st_vec = st_vec
            logging.info("spatiotemporal vector completed")

            logging.info("trajectory points and spatiotemporal_id loading...")
            stid = pd.read_csv(f"{self.data_path[: -4]}_stid.csv")
            self.stid_counts = stid.stid.value_counts().to_dict()
            logging.info("trajectory points and spatiotemporal_id load completed")

        else:
            columns = ['tid', 'time', 'lat', 'lon', 'did', 'spaceid', 'timeid', 'stid']
            self.data1 = self.data1[columns]
            self.data2 = self.data2[columns]
            test_tid = []
            test_data = {}
            for k, df in self.test.items():
                tid = df.tid.unique().tolist()
                test_tid += tid
                data1 = self.data1.query(f"tid in {tid}").copy()
                data1.reset_index(drop=True, inplace=True)
                data1.to_csv(f"{self.data_path[: -4]}_{k}_data1.csv", index=False)
                data2 = self.data2.query(f"tid in {tid}").copy()
                data2.reset_index(drop=True, inplace=True)
                data2.to_csv(f"{self.data_path[: -4]}_{k}_data2.csv", index=False)
                test_data[k] = [data1, data2]

            data1 = self.data1.query(f"tid not in {test_tid}").copy()
            data1.reset_index(drop=True, inplace=True)
            data1.to_csv(f"{self.data_path[: -4]}_train_data1.csv", index=False)
            data2 = self.data2.query(f"tid not in {test_tid}").copy()
            data2.reset_index(drop=True, inplace=True)
            data2.to_csv(f"{self.data_path[: -4]}_train_data2.csv", index=False)
            train_data = [data1, data2]

        return train_data, test_data, self.st_vec, self.stid_counts
