import logging
import re

import pandas as pd
from geopy.distance import geodesic


class GraphGenerator(object):
    def __init__(self, stid_counts):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.stid_counts = stid_counts
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def _st_vec(self, stid):
        s0, s1, t0, t1 = re.split('-+|_+', stid)
        s0, s1, t0, t1 = int(s0), int(s1), int(t0), int(t1)
        v1 = self._2binary(s0, 16)
        v2 = self._2binary(s1, 16)
        v3 = self._2binary(t0, 8)
        v4 = self._2binary(t1, 16)
        return v1 + v2 + v3 + v4

    def _2binary(self, num, length):
        bin_str = bin(num)[2:]
        if len(bin_str) < length:
            padding = '0' * (length - len(bin_str))
            bin_str = padding + bin_str
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
        # np.mean(np.array(u_vec), axis=0).tolist()      [sum(x) for x in zip(*u_vec)]
        tid_node = [sum(x) for x in zip(*tid_node)]  # 航迹代表向量
        # spatiotemporal node
        st_node = [self._st_vec(stid) for stid in stid_lis]  # stid 向量组
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
        # np.mean(np.array(u_vec), axis=0).tolist()      [sum(x) for x in zip(*u_vec)]
        tid_node = [sum(x) for x in zip(*tid_node)]  # 航迹代表向量
        # spatiotemporal node
        st_node = [self._st_vec(stid) for stid in stid_lis]  # stid 向量组
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


class GraphGenerator1(object):
    def __init__(self, st_vec, stid_counts):
        """
        tid: trajectory id
        st: spatiotemporal
        """
        self.st_vec = st_vec
        self.stid_counts = stid_counts
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def graph1(self, tid, df):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
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

    def graph2(self, tid, df):
        # get tsid 、user tsid、 node index
        node_id = {}  # tid or stid : index
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



class GraphPreprocessor(GraphGenerator):
    def __init__(self, data_path=None):
        self.data_path = data_path
        logging.info(f"self.path={self.data_path}")

    def run(self):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=16)
        logging.info("spatiotemporal vector loading...")
        st_vec = pd.read_csv(f"{self.data_path[: -4]}_st_vec.csv")
        st_vec['vec'] = st_vec['vec'].parallel_map(lambda x: eval(x))
        logging.info("spatiotemporal vector load completed")
        logging.info("trajectory points and spatiotemporal_id loading...")
        stid = pd.read_csv(f"{self.data_path[: -4]}_stid.csv")
        stid_counts = stid.stid.value_counts().to_dict()
        logging.info("trajectory points and spatiotemporal_id load completed")

        super().__init__(st_vec, stid_counts)

        dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'spaceid': str, 'timeid': str, 'stid': str}
        logging.info("generate graph1...")
        logging.info("data preparation...")
        df = pd.read_csv(f"{self.data_path[: -4]}_data1.csv", dtype=dic)
        graph1 = df[['tid']].copy()
        graph1.drop_duplicates(inplace=True)
        df = df.groupby('tid')
        logging.info("data preparation completed")
        logging.info("graph1...")
        graph1['graph'] = graph1['tid'].parallel_map(lambda x: self.graph1(x, df.get_group(x)))
        logging.info("generate graph1 completed")

        logging.info("generate graph2...")
        logging.info("data preparation...")
        df = pd.read_csv(f"{self.data_path[: -4]}_data2.csv", dtype=dic)
        graph2 = df[['tid']].copy()
        graph2.drop_duplicates(inplace=True)
        df = df.groupby('tid')
        logging.info("data preparation completed")
        logging.info("graph1...")
        graph2['graph'] = graph2['tid'].parallel_map(lambda x: self.graph2(x, df.get_group(x)))
        logging.info("generate graph1 completed")

        graph1.to_csv(f"{self.data_path[: -4]}_graph1.csv", index=False)
        graph2.to_csv(f"{self.data_path[: -4]}_graph2.csv", index=False)
        return graph1, graph2

    def load(self):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=16)
        logging.info("graph loading...")
        graph1 = pd.read_csv(f"{self.data_path[: -4]}_graph1.csv")
        graph1['graph'] = graph1['graph'].parallel_map(lambda x: eval(x))
        graph2 = pd.read_csv(f"{self.data_path[: -4]}_graph2.csv")
        graph2['graph'] = graph2['graph'].parallel_map(lambda x: eval(x))
        logging.info("graph load completed")
        return graph1, graph2

    def get(self, method="load"):
        if method == "load":
            return self.load()
        if method == "run":
            return self.run()
