import os
import pickle
import random
import re
import shutil

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from torch.utils.data import Dataset
from random import sample


class GraphGenerator(object):
    def __init__(self, stid_counts, graph_dim):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.stid_counts = stid_counts
        self.graph_dim = graph_dim
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def _st_vec(self, stid):
        s0, s1, flag, t0, t1 = re.split('-+|_+', stid)
        s0, s1, flag, t0, t1 = int(s0), int(s1), int(flag), int(t0), int(t1)
        v1 = self._2binary(s0, self.graph_dim['dim1'])  # lat
        v2 = self._2binary(s1, self.graph_dim['dim2'])  # lon
        v3 = self._2binary(flag, self.graph_dim['dim3'])  # did flag
        v4 = self._2binary(t0, self.graph_dim['dim4'])  # t0
        v5 = self._2binary(t1, self.graph_dim['dim5'])  # t1
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

        node, edge_ind, edge_attr = np.array(node), np.array(edge_ind), np.array(edge_attr)
        space_range, time_range = np.array(space_range), np.array(time_range)
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

        node, edge_ind, edge_attr = np.array(node), np.array(edge_ind), np.array(edge_attr)
        space_range, time_range = np.array(space_range), np.array(time_range)
        return node, edge_ind, edge_attr, space_range, time_range


class GraphSaver(Dataset):
    def __init__(self, path: str, tid: list, stid_counts: dict, graph_dim):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.path = path
        self.tid = tid
        self.generator = GraphGenerator(stid_counts, graph_dim)

    def __len__(self):
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        tid = self.tid[index]
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path}traj1/{tid}.csv", dtype=dic)
        node, edge_ind, edge_attr, space_range, time_range = self.generator.graph1(df1)
        np.savez(f"{self.path}graph1/{tid}.npz", node=node, edge=edge_ind,
                 edge_attr=edge_attr, space_range=space_range, time_range=time_range)

        df2 = pd.read_csv(f"{self.path}traj2/{tid}.csv", dtype=dic)
        node, edge_ind, edge_attr, space_range, time_range = self.generator.graph2(df2)
        np.savez(f"{self.path}graph2/{tid}.npz", node=node, edge=edge_ind, edge_attr=edge_attr,
                 space_range=space_range, time_range=time_range)

        return "save success"


class PSGraphSaver(GraphSaver):
    def __init__(self, path: str, tid: list, stid_counts: dict, graph_dim):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        super().__init__(path, tid, stid_counts, graph_dim)

    def _ps_df(self, tid, data):
        """
        es: enhance sample
        random: 随机删除轨迹点
        st: 随机保留 spatiotemporal相同标号 一个轨迹点
        """
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df = pd.read_csv(f"{self.path}traj{data}/{tid}.csv", dtype=dic)
        threshold = 3
        if df.shape[0] <= threshold:
            return {}

        ps = {}
        frac_list = [0.9, 0.8, 0.7, 0.6]

        # random
        random_frac_list = frac_list if df.shape[0] > 10 else frac_list[: 2]
        for i in random_frac_list:
            ps[f"{tid}_random_{i}"] = df.sample(frac=i)

        # st
        stid = df.stid.unique().tolist()
        if len(stid) < threshold:
            pass
        else:
            st_frac_list = frac_list if len(stid) > threshold * 2 else frac_list[: 2]
            for i in st_frac_list:
                sample_size = int(len(stid) * i)
                sample_size = threshold - 1 if sample_size < threshold - 1 else sample_size
                stid = random.sample(stid, sample_size)
                ps[f"{tid}_st_{i}"] = df.query(f"stid in {stid}")

        # s
        df['spaceid'] = df['stid'].map(lambda x: x.split('_')[0])
        spaceid = df.spaceid.unique().tolist()
        if len(spaceid) < threshold:
            pass
        else:
            s_frac_list = frac_list if len(spaceid) > threshold * 2 else frac_list[: 2]
            for i in s_frac_list:
                sample_size = int(len(spaceid) * i)
                sample_size = threshold - 1 if sample_size < threshold - 1 else sample_size
                spaceid = random.sample(spaceid, sample_size)
                ps[f"{tid}_s_{i}"] = df.query(f"spaceid in {spaceid}")

        # t
        df['timeid'] = df['stid'].map(lambda x: x.split('_')[1])
        timeid = df.timeid.unique().tolist()
        if len(timeid) < threshold:
            return None
        else:
            t_frac_list = frac_list if len(timeid) > threshold * 2 else frac_list[: 2]
            for i in t_frac_list:
                sample_size = int(len(timeid) * i)
                sample_size = threshold - 1 if sample_size < threshold - 1 else sample_size
                timeid = random.sample(timeid, sample_size)
                ps[f"{tid}_t_{i}"] = df.query(f"timeid in {timeid}")

        return ps

    def get_sample(self, index):
        tid = self.tid[index]
        tid_ps = {"tid": tid, "ps1": [], "ps2": []}

        ps1_dic = self._ps_df(tid, data=1)
        if ps1_dic:
            tid_ps['ps1'] = list(ps1_dic.keys())
            for name, df in ps1_dic.items():
                node, edge_ind, edge_attr, space_range, time_range = self.generator.graph1(df)
                np.savez(f"{self.path}ps_graph1/{name}.npz", node=node, edge=edge_ind, edge_attr=edge_attr,
                         space_range=space_range, time_range=time_range)

        ps2_dic = self._ps_df(tid, data=2)
        if ps2_dic:
            tid_ps['ps2'] = list(ps2_dic.keys())
            for name, df in ps2_dic.items():
                node, edge_ind, edge_attr, space_range, time_range = self.generator.graph2(df)
                np.savez(f"{self.path}ps_graph2/{name}.npz", node=node, edge=edge_ind, edge_attr=edge_attr,
                         space_range=space_range, time_range=time_range)

        return tid_ps


class NSGraphSaver(GraphSaver):
    def __init__(self, path: str, tid: list, stid_counts: dict, graph_dim):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        super().__init__(path, tid, stid_counts, graph_dim)

    def get_sample(self, index):
        tid = self.tid[index]
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}

        if "ns_traj1" in tid:
            df1 = pd.read_csv(f"{self.path}{tid}.csv", dtype=dic)
            node, edge_ind, edge_attr, space_range, time_range = self.generator.graph1(df1)
            tid = tid.replace('traj1', 'graph1')
            np.savez(f"{self.path}{tid}.npz", node=node, edge=edge_ind, edge_attr=edge_attr,
                     space_range=space_range, time_range=time_range)

        if "ns_traj2" in tid:
            df2 = pd.read_csv(f"{self.path}{tid}.csv", dtype=dic)
            node, edge_ind, edge_attr, space_range, time_range = self.generator.graph2(df2)
            tid = tid.replace('traj2', 'graph2')
            np.savez(f"{self.path}{tid}.npz",node=node, edge=edge_ind, edge_attr=edge_attr,
                     space_range=space_range, time_range=time_range)

        return "save success"


class GraphLoader(Dataset):
    def __init__(self, path: str, tid: list, train=True, enhance_tid=None, ns_num=None, ps_num=None):
        """
        tid: trajectory id
        enhance_tid: 包含A tid 和 B tid的 enhance negative sample and enhance passive sample， train=True时起作用
        ns_num: default=None 获取全部的enhance negative sample
        ps_num: default=None 获取全部的enhance passive sample
        """
        self.path = path
        self.tid = tid
        self.train = train
        self.enhance_tid = enhance_tid
        self.ns_num = ns_num
        self.ps_num = ps_num

    def __len__(self):
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def _load_graph(self, tid, folder):
        graph = np.load(f"{self.path}{folder}/{tid}.npz")
        node = graph['node'].tolist()
        edge = graph['edge'].tolist()
        attr = graph['edge_attr'].tolist()
        space_range = graph['space_range'].tolist()
        time_range = graph['time_range'].tolist()
        return node, edge, attr, space_range, time_range

    def get_sample(self, index):
        tid = self.tid[index]
        struct = {
            "tid": tid,
            "g1": self._load_graph(tid, folder="graph1"),
            "g2": self._load_graph(tid, folder="graph2")
        }
        if self.train:
            enhance_tid = self.enhance_tid[self.enhance_tid["tid"] == tid]
            for name, num, folder in zip(
                    ['ps1', 'ps2', 'ns1', 'ns2'],
                    [self.ps_num, self.ps_num, self.ns_num, self.ns_num],
                    ["ps_graph1", "ps_graph2", "ns_graph1", "ns_graph2"]):
                enhance = enhance_tid[name].values[0]
                struct[name] = [self._load_graph(i, folder=folder) for i in enhance[: num]]
        return struct
