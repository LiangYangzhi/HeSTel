import pandas as pd
from math import cos, radians
from collections import Counter
from itertools import chain
import logging
import io
import math
import time

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=32, progress_bar=True)


class Preprocessor(object):
    def __init__(self, data_path, test_path):
        self.data_path = data_path
        self.test_path = test_path
        logging.info(f"self.path={self.data_path}")

    def run(self):
        self.loader(method="run")
        self.cleaner()
        self.space2coor()  # 空间建模
        self.time2coor()  # 时间建模
        self.stid()
        self.save()
        enhance_ns = EnhanceNS(self.data_path).get(method='run')
        return self.train_tid, self.test_tid, self.stid_counts, enhance_ns

    def loader(self, method='run'):
        if method == "run":
            logging.info("data loading...")
            data = pd.read_csv(self.data_path, dtype={
                'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
                'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
            data = data[['tid', 'time', 'lat', 'lon', 'did', 'm_time', 'm_lat', 'm_lon', 'm_did']]
            data.dropna(inplace=True)
            data.drop_duplicates(inplace=True)

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

            self.test_tid = {}
            for k, v in self.test_path.items():
                tid = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
                tid = tid.tid.unique().tolist()
                self.test_tid[k] = tid
                logging.info("data load completed")

        elif method == "load":
            logging.info("loading train tid...")
            train_tid = pd.read_csv(f"{self.data_path[: -4]}_train_tid.csv", dtype={'tid': str})
            train_tid = train_tid.tid.unique().tolist()
            logging.info("loading test tid...")
            test_tid = {}
            for k, v in self.test_path.items():
                tid = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
                tid = tid.tid.unique().tolist()
                test_tid[k] = tid
            logging.info("loading stid_counts...")
            stid_counts = pd.read_csv(f"{self.data_path[: -4]}_stid.csv").stid.value_counts().to_dict()

            logging.info("loading enhance negative sample...")
            enhance_ns = EnhanceNS(self.data_path).get(method="load")
            logging.info("data load completed")
            return train_tid, test_tid, stid_counts, enhance_ns

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

    def stid(self):
        logging.info("stid, stid_counts...")
        self.data1['stid'] = self.data1.parallel_apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)
        self.data2['stid'] = self.data2.parallel_apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)
        df1 = self.data1[['stid']].copy()
        df2 = self.data2[['stid']].copy()
        self.stid_df = pd.concat([df1, df2])
        self.stid_counts = self.stid_df.stid.value_counts().to_dict()
        logging.info("stid, stid_counts completed")

    def save(self):
        logging.info("data save...")
        self.stid_df.to_csv(f"{self.data_path[: -4]}_stid.csv", index=False)
        self.stid_counts = self.stid_df.stid.value_counts().to_dict()

        logging.info("data1...")
        columns = ['tid', 'time', 'lat', 'lon', 'stid']
        data = self.data1[columns]
        data.to_csv(f"{self.data_path[: -4]}_data1.csv", index=False)
        group = data.groupby('tid')
        tid = data.tid.unique().tolist()
        for i in tid:
            df = group.get_group(i)
            df = df[["time", "lat", "lon", "stid"]]
            df.to_csv(f"{self.data_path[: -4]}_data1/{i}.csv", index=False)

        data = self.data2[columns]
        data.to_csv(f"{self.data_path[: -4]}_data2.csv", index=False)
        group = data.groupby('tid')
        tid = data.tid.unique().tolist()
        for i in tid:
            df = group.get_group(i)
            df = df[["time", "lat", "lon", "stid"]]
            df.to_csv(f"{self.data_path[: -4]}_data2/{i}.csv", index=False)

        all_tid = data[['tid']].copy()
        all_tid.drop_duplicates(inplace=True)
        test_tid = []
        for _, tid in self.test_tid.items():
            test_tid += tid
        train_tid = all_tid.query(f"tid not in {test_tid}").copy()
        train_tid.reset_index(drop=True, inplace=True)
        train_tid.to_csv(f"{self.data_path[: -4]}_train_tid.csv", index=False)
        self.train_tid = train_tid.tid.unique().tolist()
        logging.info("data save completed")

    def get(self, method="load"):
        if method == "load":
            return self.loader(method=method)
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
        df['timeid'] = df.parallel_apply(lambda row: self._create_timeid(row.tgroup, row.time), axis=1)
        return df


class EnhanceNS(object):  # Enhance negative samples
    def __init__(self, path):
        self.path = path

    def run(self):
        self.loader()
        self.generate_ns()
        return self.tid

    def get(self, method="load"):
        if method == "load":
            return self.loader(method=method)
        if method == "run":
            return self.run()

    def loader(self, method="run"):
        if method == "run":
            logging.info("EnhanceNS data preparation...")
            dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'spaceid': str, 'timeid': str, 'stid': str}
            data1 = pd.read_csv(f"{self.path[: -4]}_data1.csv", dtype=dic)
            data2 = pd.read_csv(f"{self.path[: -4]}_data2.csv", dtype=dic)
            train_tid = pd.read_csv(f"{self.path[: -4]}_train_tid.csv", dtype={'tid': str})
            self.tid = train_tid[['tid']].copy()
            self.tid.drop_duplicates(inplace=True)
            train_tid = train_tid.tid.unique().tolist()
            data1 = data1.query(f"tid in {train_tid}").copy()
            data1['spaceid'] = data1['stid'].map(lambda x: x.split('_')[0])
            data1['timeid'] = data1['stid'].map(lambda x: x.split('_')[1])
            data2 = data2.query(f"tid in {train_tid}").copy()
            data2['spaceid'] = data2['stid'].map(lambda x: x.split('_')[0])
            data2['timeid'] = data2['stid'].map(lambda x: x.split('_')[1])

            logging.info("data group by tid...")
            self.data1_group = data1.groupby('tid')
            self.data2_group = data2.groupby('tid')
            logging.info("data group by stid...")
            self.stid_tid1 = data1.groupby('stid').agg({"tid": set})
            self.stid_tid2 = data2.groupby('stid').agg({"tid": set})
            logging.info("data group by spaceid...")
            self.space_tid1 = data1.groupby('spaceid').agg({"tid": set})
            self.space_tid2 = data2.groupby('spaceid').agg({"tid": set})
            logging.info("data group by timeid...")
            self.time_tid1 = data1.groupby('timeid').agg({"tid": set})
            self.time_tid2 = data2.groupby('timeid').agg({"tid": set})
            logging.info("data preparation completed")

        elif method == "load":
            ns = pd.read_csv(f"{self.path[: -4]}_enhance_ns.csv")
            ns['ns1'] = ns['ns1'].map(lambda x: eval(x))
            ns['ns2'] = ns['ns2'].map(lambda x: eval(x))
            return ns

    def generate_ns(self):
        logging.info("generate negative sample...")
        logging.info("negative sample1...")
        self.tid['ns1'] = self.tid['tid'].map(
            lambda x: {"st": self._generate_method(x, data='1', method='st'),
                       "s": self._generate_method(x, data='1', method='s'),
                       "t": self._generate_method(x, data='1', method='t')
                       })
        logging.info("negative sample2...")
        self.tid['ns2'] = self.tid['tid'].map(
            lambda x: {"st": self._generate_method(x, data='2', method='st'),
                       "s": self._generate_method(x, data='2', method='s'),
                       "t": self._generate_method(x, data='2', method='t')
                       })
        self.tid.to_csv(f"{self.path[: -4]}_enhance_ns.csv", index=False)
        logging.info("negative sample completed")

    def _generate_method(self, tid, data='1', method='st'):
        """
        st: spatiotemporal
        s: spatial
        t: temporal
        ns: negative sample
        """
        if data == '1':
            df = self.data1_group.get_group(tid).copy()
        elif data == '2':
            df = self.data2_group.get_group(tid).copy()
        else:
            raise ValueError(f"data={data} 不在方法random_enhance中。")

        if method == 'st':
            id_list = df.stid.unique().tolist()
            if data == '1':
                tid_lis = self.stid_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.stid_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")

        elif method == 's':
            id_list = df.spaceid.unique().tolist()
            if data == '1':
                tid_lis = self.space_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.space_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")

        elif method == 't':
            id_list = df.timeid.unique().tolist()
            if data == '1':
                tid_lis = self.time_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.time_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")
        else:
            raise ValueError(f"method={method} 不在方法st_ns中。")

        tid_lis = list(chain.from_iterable(map(list, tid_lis)))
        most_counter = Counter(tid_lis).most_common(2)  # 出现最多的top2 tid
        if len(most_counter) == 1:
            return None
        if most_counter[0][0] == tid:
            ns_tid = most_counter[1][0]
        else:
            ns_tid = most_counter[0][0]
        return ns_tid

