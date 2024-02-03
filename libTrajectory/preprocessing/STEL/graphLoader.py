import logging
import random
from geopy.distance import geodesic
import torch
from torch.utils.data import DataLoader, Dataset


class IdDataset(Dataset):
    def __init__(self, data1):
        self.data1 = data1
        self.tid = self.data1.tid.unique().tolist()

    def __len__(self):
        return len(self.tid)

    def __getitem__(self, index):
        return self.tid[index]


class GraphDataset(Dataset):
    def __init__(self, data1, ts_vec, data2, st_vec):
        self.tid = data1.tid.unique().tolist()
        self.data1_group = data1.groupby('tid')
        self.ts_vec = ts_vec
        self.data2_group = data2.groupby('tid')
        self.st_vec = st_vec

    def __len__(self):
        return self.tid.__len__()

    def ts_graph(self, tid):
        data1 = self.data1_group.get_group(tid).copy()
        tsid = data1.tsid.unique().tolist()
        ts_vec = self.ts_vec.query(f"tsid in {tsid}")
        if data1.shape[0] == 1:
            node = ts_vec.vector.tolist()
            edge_ind = [[0], [0]]
            edge_attr = [1]
            return node, edge_ind, edge_attr

        node = ts_vec.vector.tolist()
        tsid_ind = {flag: ind for ind, flag in enumerate(tsid)}
        edge_ind = [[], []]
        edge_attr = []
        data1.sort_values(['time'], inplace=True)
        data1.reset_index(drop=True, inplace=True)

        data1['idts0'] = data1.apply(lambda row: (row.tsid, row.lat, row.lon, row.time), axis=1)
        data1['idts1'] = data1.groupby('tseg')['idts0'].shift(1)
        spatio = data1[~data1['idts1'].isna()].copy()   # 获取每个时间片内的空间点，计算空间点距离
        temporal = data1[data1['idts1'].isna()].copy()  # 获取每个时间片内的第一个点，计算时间片间隔
        temporal['idts1'] = temporal['idts0'].shift(1)
        temporal = temporal[~temporal['idts1'].isna()]
        if spatio.shape[0] > 1:
            spatio['dis'] = spatio.apply(
                lambda row: (row.idts0[0], row.idts1[0], geodesic(row.idts0[1:3], row.idts1[1:3]).km + 0.1), axis=1)
            for i in spatio.dis.tolist():
                edge_ind[0].append(tsid_ind[i[0]])
                edge_ind[1].append(tsid_ind[i[1]])
                edge_attr.append(i[2])

        if temporal.shape[0] > 1:
            temporal['int'] = temporal.apply(
                lambda row: (row.idts0[0], row.idts1[0], (abs(row.idts0[3] - row.idts1[3]) + 0.1) / 1000), axis=1)
            for i in temporal.int.tolist():
                edge_ind[0].append(tsid_ind[i[0]])
                edge_ind[1].append(tsid_ind[i[1]])
                edge_attr.append(i[2])
        return node, edge_ind, edge_attr

    def st_graph(self, tid):
        data2 = self.data2_group.get_group(tid).copy()
        stid = data2.stid.unique().tolist()
        st_vec = self.st_vec.query(f"stid in {stid}")
        if data2.shape[0] == 1:
            node = st_vec.vector.tolist()
            edge_ind = [[0], [0]]
            edge_attr = [1]
            return node, edge_ind, edge_attr

        node = st_vec.vector.tolist()
        stid_ind = {flag: ind for ind, flag in enumerate(stid)}
        edge_ind = [[], []]
        edge_attr = []
        data2.sort_values(['time'], inplace=True)
        data2.reset_index(drop=True, inplace=True)

        data2['idst0'] = data2.apply(lambda row: (row.stid, row.lat, row.lon, row.time), axis=1)
        data2['idst1'] = data2.groupby('sseg')['idst0'].shift(1)
        temporal = data2[~data2['idst1'].isna()].copy()  # 获取每个空间片内的时间点，计算时间点间隔
        spatio = data2[data2['idst1'].isna()].copy()  # 获取每个空间片内的第一个点，计算空间片距离
        spatio['idst1'] = spatio['idst0'].shift(1)
        spatio = spatio[~spatio['idst1'].isna()]
        if temporal.shape[0] > 1:
            temporal['inter'] = temporal.apply(
                lambda row: (row.idst0[0], row.idst1[0], (abs(row.idst0[3] - row.idst1[3]) + 0.1) / 100), axis=1)
            for i in temporal.inter.tolist():
                # edge_ind.append([stid_ind[i[0]], stid_ind[i[1]]])
                edge_ind[0].append(stid_ind[i[0]])
                edge_ind[1].append(stid_ind[i[1]])
                edge_attr.append(i[2])
        if spatio.shape[0] > 1:
            spatio['dis'] = spatio.apply(
                lambda row: (row.idst0[0], row.idst1[0], geodesic(row.idst0[1:3], row.idst1[1:3]).km + 0.1), axis=1)
            for i in spatio.dis.tolist():
                edge_ind[0].append(stid_ind[i[0]])
                edge_ind[1].append(stid_ind[i[1]])
                edge_attr.append(i[2])
        return node, edge_ind, edge_attr

    def get(self, tid):
        tid_lis = tid
        g1_node = []
        g1_edge_ind = [[], []]
        g1_edge_attr = []
        batch1 = []

        g2_node = []
        g2_edge_ind = [[], []]
        g2_edge_attr = []
        batch2 = []

        for i, tid in enumerate(tid_lis):
            node1, edge_ind1, edge_attr1 = self.ts_graph(tid)
            node2, edge_ind2, edge_attr2 = self.st_graph(tid)

            size = len(g1_node)
            edge_ind1 = [[x+size for x in sub] for sub in edge_ind1]
            g1_node += node1
            g1_edge_ind[0] += edge_ind1[0]
            g1_edge_ind[1] += edge_ind1[1]
            g1_edge_attr += edge_attr1
            for _ in range(len(node1)):
                batch1.append(i)

            size = len(g2_node)
            edge_ind2 = [[x + size for x in sub] for sub in edge_ind2]
            g2_node += node2
            g2_edge_ind[0] += edge_ind2[0]
            g2_edge_ind[1] += edge_ind2[1]
            g2_edge_attr += edge_attr2
            for _ in range(len(node2)):
                batch2.append(i)

        g1_node = torch.tensor(g1_node, dtype=torch.float32)
        g1_edge_ind = torch.tensor(g1_edge_ind, dtype=torch.long)
        g1_edge_attr = torch.tensor(g1_edge_attr)
        batch1 = torch.tensor(batch1, dtype=torch.long)

        g2_node = torch.tensor(g2_node, dtype=torch.float32)
        g2_edge_ind = torch.tensor(g2_edge_ind, dtype=torch.long)
        g2_edge_attr = torch.tensor(g2_edge_attr)
        batch2 = torch.tensor(batch2, dtype=torch.long)

        return g1_node, g1_edge_ind, g1_edge_attr, batch1, g2_node, g2_edge_ind, g2_edge_attr, batch2

    def __getitem__(self, tid):
        return self.get(tid)

