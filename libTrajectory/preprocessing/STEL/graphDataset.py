import random
import re

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from torch.utils.data import Dataset
from random import sample


class GraphGenerator(object):
    def __init__(self, path, stid_counts):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.stid_counts = stid_counts
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def _st_vec(self, stid):
        # s0, s1, t0, t1 = re.split('-+|_+', stid)
        # s0, s1, t0, t1 = int(s0), int(s1), int(t0), int(t1)
        # v1 = self._2binary(s0, 8)  # lat
        # v2 = self._2binary(s1, 10)  # lon
        # v3 = self._2binary(t0, 4)  # t0
        # v4 = self._2binary(t1, 10)  # t1
        # vec = v1 + v2 + v3 + v4
        # return np.array(vec)
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

    def graph1(self, tid, df):
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
        node_id[tid] = 0
        for i, stid in enumerate(stid_lis):
            node_id[stid] = i + 1

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        # edge: user(tid) node and spatiotemporal node
        sub_stid_counts = df.stid.value_counts().to_dict()
        for stid, count in sub_stid_counts.items():
            edge_ind[0].append(node_id[tid])
            edge_ind[1].append(node_id[stid])
            edge_attr.append(count)

        # edge between spatiotemporal node
        df.sort_values(['time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
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
        if temporal.shape[0] > 1:
            temporal['int'] = temporal.apply(
                lambda row: (row.st0[0], row.st1[0], (abs(row.st0[3] - row.st1[3]) + 0.1) / 1000), axis=1)
            for i in temporal.int.tolist():
                edge_ind[0].append(node_id[i[0]])  # i[0] tsid0
                edge_ind[1].append(node_id[i[1]])  # i[1] tsid1
                edge_attr.append(i[2])
        return node, edge_ind, edge_attr

    def graph2(self, tid, df):
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
        node_id[tid] = 0
        for i, stid in enumerate(stid_lis):
            node_id[stid] = i + 1

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        # edge: user(tid) node and spatiotemporal node
        sub_stid_counts = df.stid.value_counts().to_dict()
        for stid, count in sub_stid_counts.items():
            edge_ind[0].append(node_id[tid])
            edge_ind[1].append(node_id[stid])
            edge_attr.append(count)

        # edge between spatiotemporal node
        df.sort_values(['time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['st0'] = df.apply(lambda row: (row.stid, row.lat, row.lon, row.time), axis=1)
        df['spaceid'] = df['stid'].map(lambda x: x.split('_')[0])
        df['st1'] = df.groupby('spaceid')['st0'].shift(1)
        temporal = df[~df['st1'].isna()].copy()  # 获取每个空间片内的时间点，计算时间点间隔
        spatio = df[df['st1'].isna()].copy()  # 获取每个空间片内的第一个点，计算空间片距离
        spatio['st1'] = spatio['st0'].shift(1)
        spatio = spatio[~spatio['st1'].isna()]
        if temporal.shape[0] > 1:
            temporal['inter'] = temporal.apply(
                lambda row: (row.st0[0], row.st1[0], (abs(row.st0[3] - row.st1[3]) + 0.1) / 100), axis=1)
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
        return node, edge_ind, edge_attr


class GraphDataset(Dataset):
    def __init__(self, path: str, tid: list, stid_counts: dict, train=True, enhance_ns=None):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.path = path
        self.tid = tid
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量
        self.generator = GraphGenerator(path, stid_counts)

        self.train = train
        self.struct = None  # enhance、positive and negative sample struct
        if self.train:
            self.enhance_ns = enhance_ns

    def __len__(self):
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = self.generate_sample_tid(tid)
            for i in ['g1', 'ps1', 'ns1']:
                if struct[i] is not None:
                    tid, df1 = struct[i]
                    struct[i] = self.generator.graph1(tid, df1)
            for i in ['g2', 'ps2', 'ns2']:
                if struct[i] is not None:
                    tid, df2 = struct[i]
                    struct[i] = self.generator.graph2(tid, df2)
        else:
            dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
            df1 = pd.read_csv(f"{self.path}/data1/{tid}.csv", dtype=dic)
            df2 = pd.read_csv(f"{self.path}/data2/{tid}.csv", dtype=dic)
            node1, edge_ind1, edge_attr1 = self.generator.graph1(tid, df1)
            node2, edge_ind2, edge_attr2 = self.generator.graph2(tid, df2)
            struct = (node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2)

        return struct

    def generate_sample_tid(self, tid):
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path}/data1/{tid}.csv", dtype=dic)
        df2 = pd.read_csv(f"{self.path}/data2/{tid}.csv", dtype=dic)
        sample_tid = {
            "g1": (tid, df1),
            "g2": (tid, df2),
            "ps1": None,
            'ps2': None,
            'ns1': None,
            'ns2': None}
        # 增强正样本
        for i in ['1', '2']:
            method = sample(['random', 'st', 'st', 's', 's', 't'], 1)[0]
            en_df = self.generate_es(sample_tid[f"g{i}"][1], method=method)
            if en_df is not None:
                en_tid = f"{tid}_enhance_ps_{method}"
                sample_tid[f"ps{i}"] = (en_tid, en_df)

        # 增强负样本
        enhance_ns = self.enhance_ns.query(f"tid == '{tid}'").ns1.values[0]

        method = sample(['st', 'st', 'st', 's', 's', 't'], 1)[0]
        ns_tid = enhance_ns[method]
        if ns_tid is not None:
            df1 = pd.read_csv(f"{self.path}/data1/{ns_tid}.csv", dtype=dic)
            sample_tid["ns1"] = (ns_tid, df1)
        method = sample(['st', 'st', 'st', 's', 's', 't'], 1)[0]
        ns_tid = enhance_ns[method]
        if ns_tid is not None:
            df2 = pd.read_csv(f"{self.path}/data2/{ns_tid}.csv", dtype=dic)
            sample_tid["ns2"] = (ns_tid, df2)

        return sample_tid

    def generate_es(self, df, method='random'):
        """
        es: enhance sample
        random: 随机删除轨迹点
        st: 随机保留 spatiotemporal相同标号 一个轨迹点
        """
        if df.shape[0] <= 3:
            return None

        frac = sample([0.8, 0.7, 0.6, 0.5, 0.4], 1)[0]
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


class GraphSaver(GraphDataset):
    def __init__(self, path: str, tid: list, stid_counts: dict):
        super().__init__(path, tid, stid_counts)

    def get_sample(self, index):
        tid = self.tid[index]
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path}traj1/{tid}.csv", dtype=dic)
        node1, edge_ind1, edge_attr1 = self.generator.graph1(tid, df1)
        node1, edge_ind1, edge_attr1 = np.array(node1), np.array(edge_ind1), np.array(edge_attr1)
        np.savez(f"{self.path}graph1/{tid}.npz", node=node1, edge=edge_ind1, edge_attr=edge_attr1)

        df2 = pd.read_csv(f"{self.path}traj2/{tid}.csv", dtype=dic)
        node2, edge_ind2, edge_attr2 = self.generator.graph2(tid, df2)
        node2, edge_ind2, edge_attr2 = np.array(node2), np.array(edge_ind2), np.array(edge_attr2)
        np.savez(f"{self.path}graph2/{tid}.npz", node=node2, edge=edge_ind2, edge_attr=edge_attr2)
        return "save success"


class GraphLoader(GraphDataset):
    def __init__(self, path: str, tid: list, stid_counts, train=True, enhance_ns=None):
        super().__init__(path, tid, stid_counts, train, enhance_ns)

    def load_graph(self, tid, method="1"):
        graph = np.load(f"{self.path}graph{method}/{tid}.npz")
        node = graph['node'].tolist()
        edge = graph['edge'].tolist()
        attr = graph['edge_attr'].tolist()
        return node, edge, attr

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = self.generate_sample_tid(tid)
            for i in ['g1']:  # , 'ns1'
                if struct[i] is not None:
                    tid = struct[i][0]
                    struct[i] = self.load_graph(tid, method="1")
            for i in ['g2']:  # , 'ns2'
                if struct[i] is not None:
                    tid = struct[i][0]
                    struct[i] = self.load_graph(tid, method="2")
            # if struct['ps1'] is not None:
            #     tid, df = struct['ps1']
            #     struct['ps1'] = self.generator.graph1(tid, df)
            # if struct['ps2'] is not None:
            #     tid, df = struct['ps2']
            #     struct['ps2'] = self.generator.graph2(tid, df)
        else:
            node1, edge_ind1, edge_attr1 = self.load_graph(tid, method="1")
            node2, edge_ind2, edge_attr2 = self.load_graph(tid, method="2")
            struct = (node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2)

        return struct

    def generate_sample_tid(self, tid):
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path}traj1/{tid}.csv", dtype=dic)
        df2 = pd.read_csv(f"{self.path}traj2/{tid}.csv", dtype=dic)
        sample_tid = {
            "g1": (tid, df1),
            "g2": (tid, df2),
            "ps1": None,
            'ps2': None,
            'ns1': None,
            'ns2': None}
        # 增强正样本
        # for i in ['1', '2']:
        #     method = sample(sum([['random'] * 1, ['st'] * 3, ['s'] * 2, ['t'] * 1], []), 1)[0]
        #     en_df = self.generate_es(sample_tid[f"g{i}"][1], method=method)
        #     if en_df is not None:
        #         en_tid = f"{tid}_enhance_ps_{method}"
        #         sample_tid[f"ps{i}"] = (en_tid, en_df)

        # 增强负样本
        # df = self.enhance_ns.query(f"tid == '{tid}'")
        # enhance_ns1 = df.ns1.values[0]
        # method = sample(sum([['st'] * 3, ['s'] * 2, ['t'] * 1], []), 1)[0]
        # ns_tid = enhance_ns1[method]
        # if ns_tid is not None:
        #     sample_tid["ns1"] = (ns_tid, None)
        #
        # enhance_ns2 = df.ns2.values[0]
        # method = sample(sum([['st'] * 3, ['s'] * 2, ['t'] * 1], []), 1)[0]
        # ns_tid = enhance_ns2[method]
        # if ns_tid is not None:
        #     sample_tid["ns2"] = (ns_tid, None)

        return sample_tid
