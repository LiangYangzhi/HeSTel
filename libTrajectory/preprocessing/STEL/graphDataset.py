import pickle
import random
import re

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from torch.utils.data import Dataset
from random import sample


class GraphGenerator(object):
    def __init__(self, stid_counts):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.stid_counts = stid_counts
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def _st_vec(self, stid):
        s0, s1, flag, t0, t1 = re.split('-+|_+', stid)
        s0, s1, flag, t0, t1 = int(s0), int(s1), int(flag), int(t0), int(t1)
        v1 = self._2binary(s0, 8)  # lat
        v2 = self._2binary(s1, 10)  # lon
        v3 = self._2binary(flag, 2)  # did flag
        v4 = self._2binary(t0, 4)  # t0
        v5 = self._2binary(t1, 10)  # t1
        vec = v1 + v2 + v3 + v4 + v5
        return np.array(vec)

    def _2binary(self, num, length):
        bin_str = bin(num)[2:]
        if len(bin_str) < length:
            padding = '0' * (length - len(bin_str))
            bin_str = padding + bin_str
        elif len(bin_str) > length:
            print(f"num={num} 超出长度{length}.")
            raise print(f"num 超出长度{length}.")
        bin_vec = [int(i) for i in bin_str]
        return bin_vec

    def graph1(self, df):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
        stid_lis = df.stid.unique().tolist()
        dic = {j: self.stid_counts[j] for j in stid_lis}
        tid_stid = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]

        # tid node
        tid_node = [self._st_vec(stid) for stid in tid_stid]
        tid_node = np.mean(np.array(tid_node), axis=0).tolist()  # 航迹代表向量
        # spatiotemporal node
        st_node = [self._st_vec(stid).tolist() for stid in stid_lis]  # stid 向量组
        # node
        node = [tid_node] + st_node
        node_id["tid"] = 0
        for i, stid in enumerate(stid_lis):
            node_id[stid] = i + 1

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        # edge: user(tid) node and spatiotemporal node
        sub_stid_counts = df.stid.value_counts().to_dict()
        for stid, count in sub_stid_counts.items():
            edge_ind[0].append(node_id["tid"])
            edge_ind[1].append(node_id[stid])
            edge_attr.append(count)

        # edge between spatiotemporal node
        df.sort_values(['time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.copy()
        space_range = 0.1
        time_range = [abs(df['time'].iloc[0] - df['time'].iloc[-1] + 0.1) / 1000] * len(node)
        df['st0'] = df.apply(lambda row: (row.stid, row.lat, row.lon, row.time), axis=1)
        df['timeid'] = df['stid'].map(lambda x: x.split('_')[1])
        df['st1'] = df.groupby('timeid')['st0'].shift(1)
        spatio = df[~df['st1'].isna()].copy()  # 获取每个时间片内的空间点，计算空间点距离
        temporal = df[df['st1'].isna()].copy()  # 获取每个时间片内的第一个点，计算时间片间隔
        temporal['st1'] = temporal['st0'].shift(1)
        temporal = temporal[~temporal['st1'].isna()]
        if spatio.shape[0] > 1:
            spatio['dis'] = spatio.apply(
                lambda row: (row.st0[0], row.st1[0], geodesic(row.st0[1:3], row.st1[1:3]).km + 0.1), axis=1)
            for i in spatio.dis.tolist():
                edge_ind[0].append(node_id[i[0]])  # i[0] stid0
                edge_ind[1].append(node_id[i[1]])  # i[1] stid1
                edge_attr.append(i[2])
            dis = [i[2] for i in spatio.dis.tolist()]
            space_range += sum(dis)
        space_range = [space_range] * len(node)
        if temporal.shape[0] > 1:
            temporal['int'] = temporal.apply(
                lambda row: (row.st0[0], row.st1[0], (abs(row.st0[3] - row.st1[3]) + 0.1) / -1000), axis=1)
            for i in temporal.int.tolist():
                edge_ind[0].append(node_id[i[0]])  # i[0] tsid0
                edge_ind[1].append(node_id[i[1]])  # i[1] tsid1
                edge_attr.append(i[2])
        return node, edge_ind, edge_attr, space_range, time_range

    def graph2(self, df):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
        stid_lis = df.stid.unique().tolist()
        dic = {j: self.stid_counts[j] for j in stid_lis}
        tid_stid = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]

        # tid node
        tid_node = [self._st_vec(stid) for stid in tid_stid]
        tid_node = np.mean(np.array(tid_node), axis=0).tolist()  # 航迹代表向量
        # spatiotemporal node
        st_node = [self._st_vec(stid).tolist() for stid in stid_lis]  # stid 向量组
        # node
        node = [tid_node] + st_node
        node_id["tid"] = 0
        for i, stid in enumerate(stid_lis):
            node_id[stid] = i + 1

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        # edge: user(tid) node and spatiotemporal node
        sub_stid_counts = df.stid.value_counts().to_dict()
        for stid, count in sub_stid_counts.items():
            edge_ind[0].append(node_id["tid"])
            edge_ind[1].append(node_id[stid])
            edge_attr.append(count)

        # edge between spatiotemporal node
        df.sort_values(['time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        space_range = 0.1
        time_range = [abs(df['time'].iloc[0] - df['time'].iloc[-1] + 0.1) / 1000] * len(node)
        df = df.copy()
        df['st0'] = df.apply(lambda row: (row.stid, row.lat, row.lon, row.time), axis=1)
        df['spaceid'] = df['stid'].map(lambda x: x.split('_')[0])
        df['st1'] = df.groupby('spaceid')['st0'].shift(1)
        temporal = df[~df['st1'].isna()].copy()  # 获取每个空间片内的时间点，计算时间点间隔
        spatio = df[df['st1'].isna()].copy()  # 获取每个空间片内的第一个点，计算空间片距离
        spatio['st1'] = spatio['st0'].shift(1)
        spatio = spatio[~spatio['st1'].isna()]
        if temporal.shape[0] > 1:
            temporal['inter'] = temporal.apply(
                lambda row: (row.st0[0], row.st1[0], (abs(row.st0[3] - row.st1[3]) + 0.1) / -100), axis=1)
            for i in temporal.inter.tolist():
                edge_ind[0].append(node_id[i[0]])  # i[0] stid0
                edge_ind[1].append(node_id[i[1]])  # i[1] stid1
                edge_attr.append(i[2])
        if spatio.shape[0] > 1:
            spatio['dis'] = spatio.apply(
                lambda row: (row.st0[0], row.st1[0], geodesic(row.st0[1:3], row.st1[1:3]).km + 0.1), axis=1)
            for i in spatio.dis.tolist():
                edge_ind[0].append(node_id[i[0]])  # i[0] stid0
                edge_ind[1].append(node_id[i[1]])  # i[1] stid1
                edge_attr.append(i[2])
            dis = [i[2] for i in spatio.dis.tolist()]
            space_range += sum(dis)
        space_range = [space_range] * len(node)

        return node, edge_ind, edge_attr, space_range, time_range


class GraphSaver(Dataset):
    def __init__(self, path: str, tid: list, stid_counts: dict, train=True, enhance_ns=None):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.path = path
        self.tid = tid
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量
        self.generator = GraphGenerator(stid_counts)

        self.train = train
        if self.train:
            self.enhance_ns = enhance_ns

    def __len__(self):
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def _ps_df(self, df, method='random'):
        """
        es: enhance sample
        random: 随机删除轨迹点
        st: 随机保留 spatiotemporal相同标号 一个轨迹点
        """
        if df.shape[0] <= 3:
            return None

        frac = sample([0.8, 0.6, 0.5, 0.4], 1)[0]
        if method == 'random':
            return df.sample(frac=frac)

        elif method == 'st':
            stid = df.stid.unique().tolist()
            if len(stid) < 3:
                return None
            sample_size = int(len(stid) * frac)
            sample_size = 2 if sample_size < 2 else sample_size
            stid = random.sample(stid, sample_size)
            df = df.query(f"stid in {stid}")
            return df

        elif method == 's':
            df['spaceid'] = df['stid'].map(lambda x: x.split('_')[0])
            spaceid = df.spaceid.unique().tolist()
            if len(spaceid) < 3:
                return None
            sample_size = int(len(spaceid) * frac)
            sample_size = 2 if sample_size < 2 else sample_size
            spaceid = random.sample(spaceid, sample_size)
            df = df.query(f"spaceid in {spaceid}")
            return df

        elif method == 't':
            df['timeid'] = df['stid'].map(lambda x: x.split('_')[1])
            timeid = df.timeid.unique().tolist()
            if len(timeid) < 3:
                return None
            sample_size = int(len(timeid) * frac)
            sample_size = 2 if sample_size < 2 else sample_size
            timeid = random.sample(timeid, sample_size)
            df = df.query(f"timeid in {timeid}")
            return df

        else:
            raise ValueError(f"method={method} 不在方法random_enhance中。")

    def get_sample(self, index):
        tid = self.tid[index]
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path}traj1/{tid}.csv", dtype=dic)
        node1, edge_ind1, edge_attr1, space_range, time_range = self.generator.graph1(df1)
        node1, edge_ind1, edge_attr1 = np.array(node1), np.array(edge_ind1), np.array(edge_attr1)
        space_range, time_range = np.array(space_range), np.array(time_range)
        np.savez(f"{self.path}graph1/{tid}.npz",
                 node=node1, edge=edge_ind1, edge_attr=edge_attr1, space_range=space_range, time_range=time_range)

        df2 = pd.read_csv(f"{self.path}traj2/{tid}.csv", dtype=dic)
        node2, edge_ind2, edge_attr2, space_range, time_range = self.generator.graph2(df2)
        node2, edge_ind2, edge_attr2 = np.array(node2), np.array(edge_ind2), np.array(edge_attr2)
        space_range, time_range = np.array(space_range), np.array(time_range)
        np.savez(f"{self.path}graph2/{tid}.npz",
                 node=node2, edge=edge_ind2, edge_attr=edge_attr2, space_range=space_range, time_range=time_range)

        # enhance sample
        method = sample(sum([['random'] * 7, ['st'] * 2, ['s'] * 2, ['t'] * 1], []), 1)[0]
        ps_df1 = self._ps_df(df1, method=method)
        if ps_df1 is not None:
            node1, edge_ind1, edge_attr1, space_range, time_range = self.generator.graph1(ps_df1)
            node1, edge_ind1, edge_attr1 = np.array(node1), np.array(edge_ind1), np.array(edge_attr1)
            space_range, time_range = np.array(space_range), np.array(time_range)
            np.savez(f"{self.path}ps_graph1/{tid}.npz",
                     node=node1, edge=edge_ind1, edge_attr=edge_attr1, space_range=space_range, time_range=time_range)

        ps_df2 = self._ps_df(df2, method=method)
        if ps_df2 is not None:
            node2, edge_ind2, edge_attr2, space_range, time_range = self.generator.graph2(ps_df2)
            node2, edge_ind2, edge_attr2 = np.array(node2), np.array(edge_ind2), np.array(edge_attr2)
            space_range, time_range = np.array(space_range), np.array(time_range)
            np.savez(f"{self.path}ps_graph2/{tid}.npz",
                     node=node2, edge=edge_ind2, edge_attr=edge_attr2, space_range=space_range, time_range=time_range)

        return "save success"


class GraphLoader(Dataset):
    def __init__(self, path: str, tid: list, train=True, enhance_ns=None):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.path = path
        self.tid = tid
        self.train = train
        self.enhance_ns = enhance_ns

    def __len__(self):
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def _load_graph(self, tid, folder):
        try:
            graph = np.load(f"{self.path}{folder}/{tid}.npz")
        except FileNotFoundError:
            return None
        node = graph['node'].tolist()
        edge = graph['edge'].tolist()
        attr = graph['edge_attr'].tolist()
        space_range = graph['space_range'].tolist()
        time_range = graph['time_range'].tolist()
        return node, edge, attr, space_range, time_range

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = {
                "g1": self._load_graph(tid, folder="graph1"),  # self.graph1[tid]  self._load_graph(tid, folder="graph1")
                "ps1": self._load_graph(tid, folder="ps_graph1"),  # None self._load_graph(tid, folder="ps_graph1")
                "g2": self._load_graph(tid, folder="graph2"),  # self.graph2[tid]  self._load_graph(tid, folder="graph2")
                "ps2": self._load_graph(tid, folder="ps_graph2")  # None self._load_graph(tid, folder="ps_graph2")
            }

            ns = self.enhance_ns[self.enhance_ns["tid"] == tid]
            dic1 = ns['ns1'].values[0]
            ns_tid1 = []
            for i in dic1.values():
                if i:
                    ns_tid1.append(i)
            if not ns_tid1:
                struct['ns1'] = None
            else:
                ns_tid1 = list(set(sum(ns_tid1, [])))
                ns_tid1 = sample(ns_tid1, 2) if len(ns_tid1) > 2 else ns_tid1
                ns_graph1 = {i: self._load_graph(i, folder="graph1") for i in ns_tid1}
                struct['ns1'] = ns_graph1

            dic2 = ns['ns2'].values[0]
            ns_tid2 = []
            for i in dic2.values():
                if i:
                    ns_tid2.append(i)
            if not ns_tid2:
                struct['ns2'] = None
            else:
                ns_tid2 = list(set(sum(ns_tid2, [])))
                ns_tid2 = sample(ns_tid2, 2) if len(ns_tid2) > 2 else ns_tid2
                ns_graph2 = {i: self._load_graph(i, folder="graph2") for i in ns_tid2}
                struct['ns2'] = ns_graph2

        else:
            struct = {
                "g1": self._load_graph(tid, folder="graph1"),
                "g2": self._load_graph(tid, folder="graph2")
            }
        return struct
