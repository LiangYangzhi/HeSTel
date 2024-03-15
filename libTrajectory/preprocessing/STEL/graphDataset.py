from collections import Counter
from itertools import chain
from geopy.distance import geodesic
from torch.utils.data import Dataset
from random import sample
from libTrajectory.preprocessing.STEL.graphGenerator import GraphGenerator


class GraphDataset(Dataset):
    def __init__(self, data1, data2, stid_counts, train=True):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.tid = list(data1.tid.unique())
        self.data1_group = data1.groupby('tid')
        self.data2_group = data2.groupby('tid')
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量
        # self.generator = GraphGenerator(st_vec, stid_counts)
        self.generator = GraphGenerator(stid_counts)

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
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = self.generate_sample_tid(tid)
            struct['g1'] = self.generator.graph1(tid, self.data1_group.get_group(tid))  # (node, edge_ind, edge_attr)
            struct['g2'] = self.generator.graph2(tid, self.data2_group.get_group(tid))
            if struct['ps1'] is not None:
                tid, df = struct['ps1']
                struct['ps1'] = self.generator.graph1(tid, df)

            if struct['ps2'] is not None:
                tid, df = struct['ps2']
                struct['ps2'] = self.generator.graph2(tid, df)

            if struct['ns1'] is not None:
                tid = struct['ns1']
                struct['ns1'] = self.generator.graph1(tid, self.data1_group.get_group(tid))

            if struct['ns2'] is not None:
                tid = struct['ns2']
                struct['ns2'] = self.generator.graph2(tid, self.data2_group.get_group(tid))
            return struct
        else:
            node1, edge_ind1, edge_attr1 = self.generator.graph1(tid, self.data1_group.get_group(tid))
            node2, edge_ind2, edge_attr2 = self.generator.graph2(tid, self.data2_group.get_group(tid))
            return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2

    def generate_sample_tid(self, tid):
        sample_tid = {
            "g1": tid,
            "g2": tid,
            "ps1": None,
            'ps2': None,
            'ns1': None,
            'ns2': None}
        # 增强正样本
        for data in ['1', '2']:
            method = sample(['random', 'st', 'st', 'st', 'st', 'st'], 1)[0]
            df = self.generate_es(tid, data=data, method=method)
            if df is not None:
                en_tid = f"{tid}_enhance_ps_{method}"
                sample_tid[f"ps{data}"] = (en_tid, df)
        # 增强负样本
        for data in ['1', '2']:
            method = sample(['st', 'st', 'st', 's', 's', 't'], 1)[0]
            ns_tid = self.generate_ns(tid, data=data, method=method)
            if ns_tid is not None:
                sample_tid[f"ns{data}"] = ns_tid
        return sample_tid

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

    def generate_ns(self, tid, data='1', method='st'):
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


class GraphDataset1(Dataset):
    def __init__(self, data1, data2, st_vec, stid_counts, train=True):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.tid = list(data1.tid.unique())
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
        return self.tid.__len__()

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        tid = self.tid[index]
        if self.train:
            struct = self.generate_sample_tid(tid)
            struct['g1'] = self.graph1(tid)  # (node, edge_ind, edge_attr)
            struct['g2'] = self.graph2(tid)  # (node, edge_ind, edge_attr)
            if struct['ps1'] is not None:
                struct['ps1'] = self.graph1(struct['ps1'])
            if struct['ps2'] is not None:
                struct['ps2'] = self.graph1(struct['ps2'])
            if struct['ns1'] is not None:
                struct['ns1'] = self.graph1(struct['ns1'])
            if struct['ns2'] is not None:
                struct['ns2'] = self.graph2(struct['ns2'])
            return struct
        else:
            node1, edge_ind1, edge_attr1 = self.graph1(tid)
            node2, edge_ind2, edge_attr2 = self.graph2(tid)
            return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2

    def generate_sample_tid(self, tid):
        sample_tid = {
            "g1": tid,
            "g2": tid,
            "ps1": None,
            'ps2': None,
            'ns1': None,
            'ns2': None}
        # 增强正样本
        for data in ['1', '2']:
            method = sample(['random', 'st', 'st', 'st', 'st', 'st'], 1)[0]
            df = self.generate_es(tid, data=data, method=method)
            if df is not None:
                en_tid = f"{tid}_enhance_ps_{method}"
                sample_tid[f"ps{data}"] = [en_tid, df]
        # 增强负样本
        for data in ['1', '2']:
            method = sample(['st', 's', 'st', 's', 't'], 1)[0]
            ns_tid = self.generate_ns(tid, data=data, method=method)
            if ns_tid is not None:
                sample_tid[f"ns{data}"] = ns_tid
        return sample_tid

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

        elif method == 'st':
            if df.shape[0] == df.drop_duplicates(subset=['stid']).shape[0]:
                # 无重复的stid, 每个stid都只有一个
                return None
            # 只保留st中轨迹点frac比例
            frac = sample([0.8, 0.7, 0.6, 0.5, 0.3], 1)[0]
            df = df.sort_values(by=['stid']).reset_index(drop=True)
            df = df.groupby("stid", group_keys=False).apply(lambda x: x.sample(frac=frac))
            return df
        else:
            raise ValueError(f"method={method} 不在方法random_enhance中。")

    def generate_ns(self, tid, data='1', method='st'):
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

    def graph1(self, x):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
        if isinstance(x, list):
            tid, df = x
        else:
            tid = x
            df = self.data1_group.get_group(tid).copy()
        stid_lis = df.stid.unique().tolist()
        dic = {j: self.stid_counts[j] for j in stid_lis}
        tid_stid = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]
        st_vec = self.st_vec.query(f"stid in {stid_lis}").copy()

        # tid node
        tid_node = [st_vec.query(f"stid == '{stid}'").vec.values[0] for stid in tid_stid]
        # np.mean(np.array(u_vec), axis=0).tolist()      [sum(x) for x in zip(*u_vec)]
        tid_node = [sum(x) for x in zip(*tid_node)]  # 航迹代表向量
        # spatiotemporal node
        st_vec.set_index('stid', inplace=True)
        st_vec = st_vec.reindex(stid_lis)
        st_node = st_vec.vec.tolist()  # stid 向量组
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

    def graph2(self, x):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
        if isinstance(x, list):
            tid, df = x
        else:
            tid = x
            df = self.data2_group.get_group(tid).copy()
        stid_lis = df.stid.unique().tolist()
        dic = {j: self.stid_counts[j] for j in stid_lis}
        tid_stid = sorted(dic, key=lambda x: x[1], reverse=False)[:self.tail]
        st_vec = self.st_vec.query(f"stid in {stid_lis}").copy()

        # tid node
        tid_node = [st_vec.query(f"stid == '{stid}'").vec.values[0] for stid in tid_stid]
        # np.mean(np.array(u_vec), axis=0).tolist()      [sum(x) for x in zip(*u_vec)]
        tid_node = [sum(x) for x in zip(*tid_node)]  # 航迹代表向量
        # spatiotemporal node
        st_vec.set_index('stid', inplace=True)
        st_vec = st_vec.reindex(stid_lis)
        st_node = st_vec.vec.tolist()  # stid 向量组
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