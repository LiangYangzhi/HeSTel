import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from math import cos, radians
from collections import Counter
from itertools import chain
import logging
import math
import time
# from torch.utils.data import DataLoader
from tqdm import tqdm

# from libTrajectory.preprocessing.STEL.graphDataset import GraphSaver, NSGraphSaver, PSGraphSaver


class Preprocessor(object):
    def __init__(self, config):
        self.path = config['path']
        self.config = config['preprocessing']
        self.test_file = self.config['test']
        test_path = {}
        for k, v in self.test_file.items():
            test_path[k] = f"{self.path}{v}"
        self.test_path = test_path
        logging.info(f"self.path={self.path}")

    def run(self):
        self.loader(method="run")
        self.cleaner()
        self.space2coor()  # 空间建模
        self.time2coor()  # 时间建模
        self.stid()
        self.save()

        EnhanceNS(self.path).run()
        enhance_tid = save_graph(self.path, self.test_file, graph_dim=self.config["graph_dim"])
        return self.train_tid, self.test_tid, enhance_tid

    def loader(self, method='run'):
        if method == "run":
            logging.info("data loading...")
            columns = ['tid', 'time', 'lat', 'lon', 'did']
            arr1 = np.load(f"{self.path}data1.npy", allow_pickle=True)
            self.data1 = pd.DataFrame(arr1, columns=columns).infer_objects()
            arr2 = np.load(f"{self.path}data2.npy", allow_pickle=True)
            self.data2 = pd.DataFrame(arr2, columns=columns).infer_objects()

            self.test_tid = {}
            for k, v in self.test_path.items():
                tid = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
                tid = tid.tid.unique().tolist()
                self.test_tid[k] = tid
                logging.info("data load completed")

        elif method == "load":
            logging.info("loading train tid...")
            with open(f"{self.path}{self.config['train']}.pkl", "rb") as f:
                train_tid = pickle.load(f)
            train_tid = [f"{tid}" for tid in train_tid]

            logging.info("loading test tid...")
            test_tid = {}
            for k, v in self.test_path.items():
                tid = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
                tid = tid.tid.unique().tolist()
                test_tid[k] = tid

            logging.info("loading enhance sample...")
            try:
                enhance_tid = pd.read_csv(f"{self.path}enhance_tid.csv")
                enhance_tid['ns1'] = enhance_tid['ns1'].map(lambda x: eval(x))
                enhance_tid['ns2'] = enhance_tid['ns2'].map(lambda x: eval(x))
                enhance_tid['ps1'] = enhance_tid['ps1'].map(lambda x: eval(x))
                enhance_tid['ps2'] = enhance_tid['ps2'].map(lambda x: eval(x))
                logging.info("data load completed")
            except FileNotFoundError as e:
                logging.info(e)
                enhance_tid = None

            return train_tid, test_tid, enhance_tid

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
        distance = self.config["space_distance"]  # 120 * 1000  # m  120
        r = 6371393  # 地球半径 单位m
        lat_step = (distance / (r * cos(radians(0)))) * (180 / 3.1415926)
        lon_step = (distance / r) * (180 / 3.1415926)
        lat_len, lon_len = abs(lat1 - lat0), abs(lon1 - lon0)
        lat_size, lon_size = math.ceil(lat_len / lat_step), math.ceil(lon_len / lon_step)
        logging.info(f"lat_len: {lat_len}, lon_len： {lon_len}， lat_step: {lat_step}, lon_step: {lon_step}")

        # 生成区域节点的lat和lon值
        lat_lis, lon_lis = [lat0], [lon0]
        for _ in range(lat_size):
            lat_lis.append(round(lat_lis[-1] + lat_step, deci))
        for _ in range(lon_size):
            lon_lis.append(round(lon_lis[-1] + lon_step, deci))
        lat_lis[-1], lon_lis[-1] = lat1, lon1
        logging.info(f"lat_lis len:{len(lat_lis)}, lon_lis len:{len(lon_lis)}")

        logging.info("data1 space coordinate...")
        space_num = SpaceNum(lat_lis, lon_lis)
        data1 = self.data1.copy()
        self.data1 = space_num.get(data1)
        self.data1['spaceid'] = self.data1['spaceid'].map(lambda x: f"{x}-0")
        logging.info("data1 space coordinate completed")
        logging.info("data2 space coordinate...")

        data2 = self.data2.copy()
        self.data2 = space_num.get(data2)
        self._space_seg()
        logging.info("data2 space coordinate completed")

    def _space_seg(self):
        did_df = self.data2[['did', 'spaceid']].copy()
        did_df.drop_duplicates(inplace=True)

        did_dup = did_df[did_df.duplicated(subset='spaceid', keep=False)]
        if did_dup.shape[0] == 0:
            self.data2['spaceid'] = self.data2['spaceid'].map(lambda x: f"{x}-0")
        else:
            didid_list = did_dup.lonid.unique().tolist()
            did_flage = {}
            for i in didid_list:
                df = did_dup[did_dup['did'] == i]
                did_list = df.did.unique().tolist()
                for num, did in enumerate(did_list):
                    did_flage[did] = num + 1
            logging.info(f"did_flage: {did_flage}")
            self.data2['spaceid'] = self.data2.apply(
                lambda row: f"{row.spaceid}-{did_flage[row.did]}" if row.did in did_flage else f"{row.spaceid}-0", axis=1)

    def time2coor(self, groupby='month'):
        logging.info("time coordinate...")
        interval = self.config['time_interval']  # 6 * 60 * 60
        logging.info(f"time interval={interval}s")
        # if groupby == 'month':
        if 'ais' in self.path:
            self.data1['tgroup'] = self.data1['time'].map(lambda t: time.localtime(t).tm_mon)
            self.data2['tgroup'] = self.data2['time'].map(lambda t: time.localtime(t).tm_mon)
            logging.info("time coordinate group by month")
        # if groupby == 'week':  # 一年内第几周
        if 'taxi' in self.path:
            self.data1['tgroup'] = self.data1['time'].map(lambda t: datetime.fromtimestamp(t).isocalendar()[1])
            self.data2['tgroup'] = self.data2['time'].map(lambda t: datetime.fromtimestamp(t).isocalendar()[1])
            # self.data1['tgroup'] = self.data1['time'].map(lambda t: time.localtime(t).tm_mday + 1)
            # self.data2['tgroup'] = self.data2['time'].map(lambda t: time.localtime(t).tm_mday + 1)
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
        time_len = 0
        for _, v in time_dict.items():
            time_len += len(v)
        logging.info(f"time_dict len:{time_len}")
        timeid = TimeNum(time_dict)
        test = self.data1.copy()
        self.data1 = timeid.get(test)
        logging.info("data1 time coordinate completed")
        logging.info("data2 time coordinate...")
        test = self.data2.copy()
        self.data2 = timeid.get(test)
        # self.data2['timeid'] = self.data2['time'].map(
        #     lambda x: f"{time.localtime(x).tm_mon}-{time.localtime(x).tm_mday * (time.localtime(x).tm_hour // 6 + 1)}")
        logging.info("data2 time coordinate completed")

    def stid(self):
        logging.info("stid, stid_counts...")
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=16, progress_bar=True)
        self.data1['stid'] = self.data1.parallel_apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)
        self.data2['stid'] = self.data2.parallel_apply(lambda row: f"{row.spaceid}_{row.timeid}", axis=1)
        df1 = self.data1[['stid']].copy()
        df2 = self.data2[['stid']].copy()
        self.stid_df = pd.concat([df1, df2])
        self.stid_counts = self.stid_df.stid.value_counts().to_dict()
        logging.info("stid, stid_counts completed")

    def save(self):
        logging.info("data save...")
        self.stid_df.to_csv(f"{self.path}stid.csv", index=False)
        self.stid_counts = self.stid_df.stid.value_counts().to_dict()
        with open(f"{self.path}stid_counts.pkl", 'wb') as f:
            pickle.dump(self.stid_counts, f)

        logging.info("data1...")
        columns = ['tid', 'time', 'lat', 'lon', 'stid']
        data = self.data1[columns]
        arr = data.to_numpy()
        np.save(f'{self.path}traj1.npy', arr)
        group = data.groupby('tid')
        tid = data.tid.unique().tolist()
        dir_path = f"{self.path}traj1"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i in tid:
            df = group.get_group(i)
            df = df[["time", "lat", "lon", "stid"]]
            df.to_csv(f"{dir_path}/{i}.csv", index=False)

        data = self.data2[columns]
        arr = data.to_numpy()
        np.save(f'{self.path}traj2.npy', arr)
        group = data.groupby('tid')
        tid = data.tid.unique().tolist()
        dir_path = f"{self.path}traj2"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i in tid:
            df = group.get_group(i)
            df = df[["time", "lat", "lon", "stid"]]
            df.to_csv(f"{dir_path}/{i}.csv", index=False)

        all_tid = data[['tid']].copy()
        all_tid.drop_duplicates(inplace=True)
        test_tid = []
        for _, tid in self.test_tid.items():
            test_tid += tid
        train_tid = all_tid.query(f"tid not in {test_tid}").copy()
        train_tid.reset_index(drop=True, inplace=True)
        self.train_tid = train_tid.tid.unique().tolist()
        with open(f"{self.path}train_tid.pkl", 'wb') as f:
            pickle.dump(self.train_tid, f)
        small_train_tid = random.sample(self.train_tid, 6000)
        with open(f"{self.path}small_train_tid.pkl", 'wb') as f:
            pickle.dump(small_train_tid, f)
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
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=32, progress_bar=True)
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
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        df['timeid'] = df.parallel_apply(lambda row: self._create_timeid(row.tgroup, row.time), axis=1)
        return df


class EnhanceNS(object):  # Enhance negative samples
    def __init__(self, path):
        self.path = path

    def run(self):
        self.loader()
        NSMultiRun(self.path).generate_ns()
        return self.tid

    def loader(self):
        logging.info("EnhanceNS data preparation...")
        columns = ['tid', 'time', 'lat', 'lon', 'stid']
        data1 = np.load(f"{self.path}traj1.npy", allow_pickle=True)
        data1 = pd.DataFrame(data1, columns=columns).infer_objects()
        data2 = np.load(f"{self.path}traj2.npy", allow_pickle=True)
        data2 = pd.DataFrame(data2, columns=columns).infer_objects()

        with open(f"{self.path}train_tid.pkl", "rb") as f:
            train_tid = pickle.load(f)
        train_tid = [f"{tid}" for tid in train_tid]
        self.tid = pd.DataFrame(data=train_tid, columns=['tid'])
        self.tid.drop_duplicates(inplace=True)

        data1 = data1.query(f"tid in {train_tid}").copy()
        data2 = data2.query(f"tid in {train_tid}").copy()

        logging.info("data group by tid...")
        global data1_group
        data1_group = data1.groupby('tid')
        global data2_group
        data2_group = data2.groupby('tid')
        logging.info("data group by stid...")
        global stid_tid1
        stid_tid1 = data1.groupby('stid').agg({"tid": set})
        global stid_tid2
        stid_tid2 = data2.groupby('stid').agg({"tid": set})
        logging.info("data preparation completed")


class NSMultiRun(object):
    def __init__(self, path):
        self.path = path
        with open(f"{self.path}train_tid.pkl", "rb") as f:
            train_tid = pickle.load(f)
        train_tid = [f"{tid}" for tid in train_tid]
        self.tid = pd.DataFrame(data=train_tid, columns=['tid'])
        self.tid.drop_duplicates(inplace=True)

    def generate_ns(self):
        logging.info("generate negative sample...")
        logging.info("negative sample1...")
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=12, progress_bar=True)
        for i in [1, 2]:
            dir_path = f"{self.path}ns_traj{i}/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        self.tid['ns1'] = self.tid['tid'].parallel_map(lambda x: self._generate_method(x, data='1'))
        logging.info("negative sample2...")
        self.tid['ns2'] = self.tid['tid'].parallel_map(lambda x: self._generate_method(x, data='2'))
        self.tid.to_csv(f"{self.path}enhance_ns.csv", index=False)
        logging.info("negative sample completed")

    def _generate_method(self, tid, data='1'):
        """ns: negative sample"""
        if data == '1':
            df = data1_group.get_group(tid).copy()
        elif data == '2':
            df = data2_group.get_group(tid).copy()
        else:
            raise ValueError(f"data={data} 不在方法random_enhance中。")
        id_list = df["stid"].unique().tolist()

        if data == '1':
            tid_lis = stid_tid1.loc[id_list, :].tid.tolist()
        elif data == '2':
            tid_lis = stid_tid2.loc[id_list, :].tid.tolist()
        else:
            raise ValueError(f"data={data} 不在方法st_ns中。")

        top = 16  # 最多有多少个ns_tid  最少是top/4
        tid_lis = list(chain.from_iterable(map(list, tid_lis)))
        most_counter = Counter(tid_lis).most_common(top + 1)  # 出现最多的top tid  起码要在三个点以上重合
        if len(most_counter) == 1:
            return []
        candidate_ns = [i[0] for i in most_counter if i[0] != tid]

        hold_flag = []
        if len(candidate_ns) <= int(top / 2):
            hold_flag = [1 for _ in range(len(candidate_ns))]
        else:
            for ns in candidate_ns:
                if data == '1':
                    df = data1_group.get_group(ns).copy()
                elif data == '2':
                    df = data2_group.get_group(ns).copy()
                ns_id_list = df["stid"].unique().tolist()
                comm_id = list(set(id_list).intersection(set(ns_id_list)))
                if len(comm_id) >= int(math.ceil(len(id_list) * 0.3)) and len(comm_id) >= 3:
                    hold_flag.append(1)
                    continue
                hold_flag.append(0)

            hold_sum = sum(hold_flag)
            if hold_sum < int(top/4):
                for i in range(int(top/4)):
                    hold_flag[i] = 1

        frac = 0.3  # diff id_list 的轨迹点保留比例
        dir_path = f"{self.path}ns_traj{data}/"
        ns_tid = []
        for i in range(len(candidate_ns)):
            if hold_flag[i]:
                ns = candidate_ns[i]
                ns_tid.append(ns)
                if data == '1':
                    df = data1_group.get_group(ns).copy()
                elif data == '2':
                    df = data2_group.get_group(ns).copy()
                same = df[df["stid"].isin(id_list)].copy()
                same_stid = same.stid.unique().tolist()
                diff = df[~df["stid"].isin(id_list)].copy()
                diff_stid = diff.stid.unique().tolist()
                if len(diff_stid) > 10:
                    value = int(math.ceil(frac*(len(diff_stid) - 5)))
                    k = 10 if value > 10 else value
                    diff_stid = random.sample(diff_stid, k)
                    diff = diff[diff['stid'].isin(diff_stid)].copy()
                if len(same_stid) > 15:
                    value = int(math.ceil(frac * 2 * (len(same_stid) - 5)))
                    k = 15 if value > 15 else value
                    same_stid = random.sample(same_stid, k)
                    same = same[same['stid'].isin(same_stid)].copy()
                df = pd.concat([same, diff])
                df.sort_values(['time'], inplace=True)
                df.to_csv(f"{dir_path}{tid}_st_{ns}.csv", index=False)
        ns_tid = [f"{tid}_st_{ns}" for ns in ns_tid]
        return ns_tid


def save_graph(path, test_file, graph_dim):
    # tid graph
    train_tid, test_tid, _ = Preprocessor(path, test_file, {}).get(method='load')
    with open(f"{path}stid_counts.pkl", "rb") as f:
        stid_counts = pickle.load(f)
    tid = train_tid.copy()
    for _, v in test_tid.items():
        tid += v
    for i in [1, 2]:
        dir_path = f"{path}graph{i}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    graph_data = GraphSaver(path, tid, stid_counts, graph_dim)
    data_loader = DataLoader(dataset=graph_data, batch_size=4, num_workers=36, persistent_workers=True)
    for _ in tqdm(data_loader):  # 每个批次循环
        pass

    # ps graph
    tid, _, _ = Preprocessor(path, test_file, {}).get(method='load')
    with open(f"{path}stid_counts.pkl", "rb") as f:
        stid_counts = pickle.load(f)
    for i in [1, 2]:
        dir_path = f"{path}ps_graph{i}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    graph_data = PSGraphSaver(path, tid, stid_counts, graph_dim)
    data_loader = DataLoader(dataset=graph_data, batch_size=4, num_workers=36,
                             persistent_workers=True, collate_fn=lambda x: x)
    ps = []
    for lis in tqdm(data_loader):  # 每个批次循环
        for dic in lis:
            ps.append(dic)
    ps = pd.DataFrame(ps)
    ps.to_csv(f"{path}enhance_ps.csv", index=False)

    # ns graph
    ns = pd.read_csv(f"{path}enhance_ns.csv")
    ns['ns1'] = ns['ns1'].map(lambda x: eval(x))
    tid1 = sum(ns.ns1.values.tolist(), [])
    tid1 = [f"ns_traj1/{i}" for i in tid1]
    ns['ns2'] = ns['ns2'].map(lambda x: eval(x))
    tid2 = sum(ns.ns2.values.tolist(), [])
    tid2 = [f"ns_traj2/{i}" for i in tid2]
    tid = sum([tid1, tid2], [])
    with open(f"{path}stid_counts.pkl", "rb") as f:
        stid_counts = pickle.load(f)
    for i in [1, 2]:
        dir_path = f"{path}ns_graph{i}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    graph_data = NSGraphSaver(path, tid, stid_counts, graph_dim)
    data_loader = DataLoader(dataset=graph_data, batch_size=4, num_workers=36, persistent_workers=True)
    for _ in tqdm(data_loader):  # 每个批次循环
        pass

    # ns_tid
    ns = pd.read_csv(f"{path}enhance_ns.csv")
    ns['ns1'] = ns['ns1'].map(lambda x: eval(x))
    ns['ns2'] = ns['ns2'].map(lambda x: eval(x))
    ps = pd.read_csv(f"{path}enhance_ps.csv")
    ps['ps1'] = ps['ps1'].map(lambda x: eval(x))
    ps['ps2'] = ps['ps2'].map(lambda x: eval(x))
    enhance_tid = pd.merge(ps, ns, on='tid', how='outer')
    enhance_tid.to_csv(f"{path}enhance_tid.csv", index=False)

    return enhance_tid
