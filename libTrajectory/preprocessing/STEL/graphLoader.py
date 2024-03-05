from collections import Counter
from itertools import chain

from geopy.distance import geodesic
import torch
from torch.utils.data import Dataset
from random import sample


class IdDataset(Dataset):
    def __init__(self, data1):
        self.tid = list(data1.tid.unique())

    def __len__(self):
        return len(self.tid)

    def __getitem__(self, index):
        return self.tid[index]


class GraphDataset(Dataset):
    def __init__(self, data1, data2, st_vec, stid_counts, train=True):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.data1_group = data1.groupby('tid')
        self.data2_group = data2.groupby('tid')
        self.st_vec = st_vec
        self.stid_counts = stid_counts
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

        self.train = train
        self.struct = None  # enhance、positive and negative sample struct
        self.g_tid_df = {"data1": {}, "data2": {}}
        if train:
            self.stid_tid1 = data1.groupby('stid').agg({"tid": set})
            self.stid_tid2 = data2.groupby('stid').agg({"tid": set})
            self.space_tid1 = data1.groupby('spaceid').agg({"tid": set})
            self.space_tid2 = data2.groupby('spaceid').agg({"tid": set})
            self.time_tid1 = data1.groupby('timeid').agg({"tid": set})
            self.time_tid2 = data2.groupby('timeid').agg({"tid": set})

    def __len__(self):
        return self.data1_group.ngroups

    def __getitem__(self, index):
        return self.get_sample(index)

    def generate_es(self, tid, data='1', method='random'):
        """
        es: enhance sample
        random: 随机删除轨迹点
        st: 随机保留 spatiotemporal相同标号 一个轨迹点
        """
        if data == '1':
            df = self.data1_group.get_group(tid).copy()
        elif data == '2':
            df = self.data2_group.get_group(tid).copy()
        else:
            raise ValueError(f"data={data} 不在方法random_enhance中。")

        if df.shape[0] <= 2:
            return None

        if method == 'random':
            return df.sample(frac=0.8)
        elif method == 'st':  # 只保留st中一个轨迹点
            if df.shape[0] == df.drop_duplicates(subset=['stid']).shape[0]:
                # 无重复的stid
                return None
            df = df.sort_values(by=['stid']).reset_index(drop=True)
            df = df.groupby("stid", group_keys=False).apply(lambda x: x.sample(1))
            return df
        else:
            raise ValueError(f"method={method} 不在方法random_enhance中。")

    def generate_ns(self, tid, data='1', method='st'):
        """
        正样本:A-B(tid相同), A的负样本从B中生成
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

        # tid_lis = sum(tid_lis, [])
        tid_lis = list(chain.from_iterable(map(list, tid_lis)))
        most_counter = Counter(tid_lis).most_common(2)  # 出现最多的top2 tid
        if len(most_counter) == 1:
            return None
        if most_counter[0][0] == tid:
            ns_tid = most_counter[1][0]
        else:
            ns_tid = most_counter[0][0]
        return ns_tid

    def graph1(self, tid_lis):
        # get tsid 、user tsid、 node index
        stid_lis = []
        tid_vec_lis = []
        node_id = {}  # tid or stid : index
        tid_df = {}  # tid: df
        for i, tid in enumerate(tid_lis):
            if 'generate' in tid:
                df = self.g_tid_df['data1'][tid]
            else:
                df = self.data1_group.get_group(tid).copy()
            stid = df.stid.unique().tolist()
            dic = {j: self.stid_counts[j] for j in stid}
            user = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]

            stid_lis.extend(stid)
            tid_vec_lis.append(user)
            node_id[tid] = i
            tid_df[tid] = df

        stid_lis = list(set(stid_lis))
        st_vec = self.st_vec.query(f"stid in {stid_lis}").copy()
        tid_len = len(tid_lis)
        for i, j in enumerate(stid_lis):
            node_id[j] = i + tid_len

        # get node
        tid_node = []
        for u in tid_vec_lis:
            u_vec = [st_vec.query(f"stid == '{i}'").vec.values[0] for i in u]
            tid_node.append([sum(x) for x in zip(*u_vec)])  # user(tid) node
        # spatiotemporal node
        st_vec.set_index('stid', inplace=True)
        st_vec = st_vec.reindex(stid_lis)
        st_node = st_vec.vec.tolist()

        node = tid_node + st_node


        # get edge
        edge_ind = [[], []]
        edge_attr = []
        for tid, df in tid_df.items():
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

        node = torch.tensor(node, dtype=torch.float32)
        edge_ind = torch.tensor(edge_ind, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return node, edge_ind, edge_attr

    def graph2(self, tid_lis):
        # get tsid 、user tsid、 node index
        stid_lis = []
        tid_vec_lis = []
        node_id = {}  # tid or stid : index
        tid_df = {}  # tid: df
        for i, tid in enumerate(tid_lis):
            if 'generate' in tid:
                df = self.g_tid_df['data2'][tid]
            else:
                df = self.data2_group.get_group(tid).copy()
            stid = df.stid.unique().tolist()
            dic = {j: self.stid_counts[j] for j in stid}
            user = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]

            stid_lis.extend(stid)
            tid_vec_lis.append(user)
            node_id[tid] = i
            tid_df[tid] = df

        stid = list(set(stid_lis))
        st_vec = self.st_vec.query(f"stid in {stid}")
        tid_len = len(tid_lis)
        for i, j in enumerate(stid):
            node_id[j] = i + tid_len

        # get node
        tid_node = []
        for u in tid_vec_lis:
            u_vec = [st_vec.query(f"stid == '{i}'").vec.values[0] for i in u]
            tid_node.append([sum(x) for x in zip(*u_vec)])  # user(tid) node
        # spatiotemporal node
        st_vec.set_index('stid', inplace=True)
        st_vec = st_vec.reindex(stid_lis)
        st_node = st_vec.vec.tolist()
        node = tid_node + st_node

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        for tid, df in tid_df.items():
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

        node = torch.tensor(node, dtype=torch.float32)
        edge_ind = torch.tensor(edge_ind, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return node, edge_ind, edge_attr

    def sample_structure(self, tid_list):
        struct = {"tid_ind1": {},
                  "tid_ind2": {},
                  "enh1": [],  # 增强样本1  [[A A'], ...]
                  'enh2': [],  # 增强样本2  [[B B'], ...]
                  'ps': [],  # 正样本对  [[A B], ...]
                  'ns1': [],  # 负样本对1 [[A B], ...]
                  'ns2': [],  # 负样本对2  [[B A], ...]
                  }

        # 正样本对 和 初始化tid_index
        for i, tid in enumerate(tid_list):
            struct['tid_ind1'][tid] = i
            struct['tid_ind2'][tid] = i
            struct['ps'].append([i, i])

        # 增强样本
        for i, tid in enumerate(tid_list):
            for data in ['1', '2']:
                method = sample(['random', 'st'], 1)[0]
                df = self.generate_es(tid, data=data, method=method)
                if df is not None:
                    en_tid = f"{tid}_generate_enhance_{method}"
                    self.g_tid_df[f"data{data}"][en_tid] = df
                    en_index = len(struct[f'tid_ind{data}'])  # 索引从0开始
                    struct[f'tid_ind{data}'][en_tid] = en_index
                    struct[f"enh{data}"].append([i, en_index])

        # 负样本
        # st: spatiotemporal
        # s: spatial
        # t: temporal
        original_ind = [i for i in range(len(tid_list))]
        for i, tid in enumerate(tid_list):
            method = sample(['st', 's', 't'], 1)[0]
            # A的负样本, 正样本A-B(tid相同), A的负样本从B中生成, 返回负样本B
            # struct["ns1"]: [A: B]
            ns_tid2 = self.generate_ns(tid, data='1', method=method)
            if ns_tid2 is not None:
                ns_index = struct['tid_ind2'].get(ns_tid2, None)
                if ns_index:  # ns_tid2已经存在tid_list中
                    struct["ns1"].append([i, ns_index])
                else:
                    ns_index = len(struct["tid_ind2"])  # 索引从0开始
                    struct['tid_ind2'][ns_tid2] = ns_index  # 索引从0开始
                    struct["ns1"].append([i, ns_index])

            # B的负样本, 正样本A-B(tid相同), B的负样本从A中生成, 返回负样本A
            # struct["ns2"]: [B: A]
            method = sample(['st', 's', 't'], 1)[0]
            ns_tid1 = self.generate_ns(tid, data='2', method=method)
            if ns_tid1 is not None:
                ns_index = struct['tid_ind1'].get(ns_tid1, None)
                if ns_index:  # ns_tid1已经存在tid_list中
                    struct["ns2"].append([i, ns_index])
                else:
                    ns_index = len(struct["tid_ind1"])  # 索引从0开始
                    struct['tid_ind1'][ns_tid1] = ns_index  # 索引从0开始
                    struct["ns2"].append([i, ns_index])

            # 随机负样本
            while True:
                random_ns = sample(original_ind, 1)[0]
                if random_ns != i:
                    break
            struct["ns2"].append([i, random_ns])
            struct["ns1"].append([random_ns, i])

        return struct

    def get_sample(self, index):
        if self.train:
            struct = self.sample_structure(index)
            tid1 = list(struct['tid_ind1'].keys())
            tid2 = list(struct['tid_ind2'].keys())
            node1, edge_ind1, edge_attr1 = self.graph1(tid1)
            node2, edge_ind2, edge_attr2 = self.graph2(tid2)
            return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2, struct
        else:
            node1, edge_ind1, edge_attr1 = self.graph1(index)
            node2, edge_ind2, edge_attr2 = self.graph2(index)
            return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2
