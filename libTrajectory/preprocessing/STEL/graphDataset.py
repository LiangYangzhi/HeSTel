from collections import Counter
from itertools import chain

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
        self.generator = GraphGenerator(stid_counts)

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
            for i in ['g1', 'ps1', 'ns1',]:
                if struct[i] is not None:
                    tid, df1 = struct[i]
                    struct[i] = self.generator.graph1(tid, df1)
            for i in ['g2', 'ps2', 'ns2']:
                if struct[i] is not None:
                    tid, df2 = struct[i]
                    struct[i] = self.generator.graph2(tid, df2)
        else:
            dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
            df1 = pd.read_csv(f"{self.path[: -4]}_data1/{tid}.csv", dtype=dic)
            df2 = pd.read_csv(f"{self.path[: -4]}_data2/{tid}.csv", dtype=dic)
            node1, edge_ind1, edge_attr1 = self.generator.graph1(tid, df1)
            node2, edge_ind2, edge_attr2 = self.generator.graph2(tid, df2)
            struct = (node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2)

        return struct

    def generate_sample_tid(self, tid):
        dic = {'time': int, 'lat': float, 'lon': float, 'stid': str}
        df1 = pd.read_csv(f"{self.path[: -4]}_data1/{tid}.csv", dtype=dic)
        df2 = pd.read_csv(f"{self.path[: -4]}_data2/{tid}.csv", dtype=dic)
        sample_tid = {
            "g1": (tid, df1),
            "g2": (tid, df2),
            "ps1": None,
            'ps2': None,
            'ns1': None,
            'ns2': None}
        # 增强正样本
        for i in ['1', '2']:
            method = sample(['random', 'st', 'st', 'st', 'st', 'st'], 1)[0]
            en_df = self.generate_es(sample_tid[f"g{i}"][1], method=method)
            if en_df is not None:
                en_tid = f"{tid}_enhance_ps_{method}"
                sample_tid[f"ps{i}"] = (en_tid, en_df)

        # 增强负样本
        enhance_ns = self.enhance_ns.query(f"tid == '{tid}'").ns1.values[0]

        method = sample(['st', 'st', 'st', 's', 's', 't'], 1)[0]
        ns_tid = enhance_ns[method]
        if ns_tid is not None:
            df1 = pd.read_csv(f"{self.path[: -4]}_data1/{ns_tid}.csv", dtype=dic)
            sample_tid["ns1"] = (ns_tid, df1)
        method = sample(['st', 'st', 'st', 's', 's', 't'], 1)[0]
        ns_tid = enhance_ns[method]
        if ns_tid is not None:
            df2 = pd.read_csv(f"{self.path[: -4]}_data2/{ns_tid}.csv", dtype=dic)
            sample_tid["ns2"] = (ns_tid, df2)

        return sample_tid

    def generate_es(self, df, method='random'):
        """
        es: enhance sample
        random: 随机删除轨迹点
        st: 随机保留 spatiotemporal相同标号 一个轨迹点
        """
        if df.shape[0] <= 2:
            return None

        if method == 'random':
            return df.sample(frac=0.8)

        elif method == 'st':
            if df.shape[0] == df.drop_duplicates(subset=['stid']).shape[0]:
                # 无重复的stid, 每个stid都只有一个
                return None
            # 只保留st中轨迹点frac比例
            df = df.sort_values(by=['stid']).reset_index(drop=True)
            frac = sample([0.8, 0.4, 0], 1)[0]
            if frac == 0:
                df = df.groupby("stid", group_keys=False).apply(lambda x: x.sample(1))
            else:
                df = df.groupby("stid", group_keys=False).apply(lambda x: x.sample(frac=frac))
            return df
        else:
            raise ValueError(f"method={method} 不在方法random_enhance中。")
