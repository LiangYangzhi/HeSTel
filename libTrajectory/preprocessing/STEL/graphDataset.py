import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from random import sample
from libTrajectory.preprocessing.STEL.graphGenerator import GraphGenerator


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
        df1 = pd.read_csv(f"{self.path}data1/{tid}.csv", dtype=dic)
        df2 = pd.read_csv(f"{self.path}data2/{tid}.csv", dtype=dic)
        node1, edge_ind1, edge_attr1 = self.generator.graph1(tid, df1)
        node2, edge_ind2, edge_attr2 = self.generator.graph2(tid, df2)
        node1, edge_ind1, edge_attr1 = np.array(node1), np.array(edge_ind1), np.array(edge_attr1)
        node2, edge_ind2, edge_attr2 = np.array(node2), np.array(edge_ind2), np.array(edge_attr2)
        np.save(f"{self.path}graph1_node/{tid}.npy", node1)
        np.save(f"{self.path}graph1_edge/{tid}.npy", edge_ind1)
        np.save(f"{self.path}graph1_attr/{tid}.npy", edge_attr1)
        np.save(f"{self.path}graph2_node/{tid}.npy", node2)
        np.save(f"{self.path}graph2_edge/{tid}.npy", edge_ind2)
        np.save(f"{self.path}graph2_attr/{tid}.npy", edge_attr2)
        return "save success"


class GraphLoader(GraphDataset):
    def __init__(self, path: str, tid: list, stid_counts, train=True, enhance_ns=None):
        super().__init__(path, tid, stid_counts, train, enhance_ns)

    def load_graph(self, tid, method="1"):
        node = np.load(f"{self.path}/graph{method}_node/{tid}.npy").tolist()
        edge = np.load(f"{self.path}/graph{method}_edge/{tid}.npy").tolist()
        attr = np.load(f"{self.path}/graph{method}_attr/{tid}.npy").tolist()
        return node, edge, attr

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = self.generate_sample_tid(tid)
            for i in ['g1', 'ns1']:  # , 'ns1'
                if struct[i] is not None:
                    tid = struct[i][0]
                    struct[i] = self.load_graph(tid, method="1")
            for i in ['g2', 'ns2']:  # , 'ns2'
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
        df1 = pd.read_csv(f"{self.path}data1/{tid}.csv", dtype=dic)
        df2 = pd.read_csv(f"{self.path}data2/{tid}.csv", dtype=dic)
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
        df = self.enhance_ns.query(f"tid == '{tid}'")
        enhance_ns1 = df.ns1.values[0]
        method = sample(sum([['st'] * 3, ['s'] * 2, ['t'] * 1], []), 1)[0]
        ns_tid = enhance_ns1[method]
        if ns_tid is not None:
            sample_tid["ns1"] = (ns_tid, None)

        enhance_ns2 = df.ns2.values[0]
        method = sample(sum([['st'] * 3, ['s'] * 2, ['t'] * 1], []), 1)[0]
        ns_tid = enhance_ns2[method]
        if ns_tid is not None:
            sample_tid["ns2"] = (ns_tid, None)

        return sample_tid
