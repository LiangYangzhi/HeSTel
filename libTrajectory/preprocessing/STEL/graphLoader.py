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
    def __init__(self, data1, ts_vec, tsid_counts, data2, st_vec, stid_counts, device):
        self.data1_group = data1.groupby('tid')
        self.ts_vec = ts_vec
        self.tsid_counts = tsid_counts
        self.data2_group = data2.groupby('tid')
        self.st_vec = st_vec
        self.stid_counts = stid_counts

        self.tid = data1.tid.unique().tolist()
        self.device = device
        self.tail = 3  # 取用户到访区域中拥有全量轨迹点最后几名作为用户向量

    def __len__(self):
        return self.tid.__len__()

    def ts_graph(self, tid_lis):
        # get tsid 、user tsid、 node index
        tsid_lis = []
        user_lis = []
        node_id = {}  # tid or tsid: ind
        tid_df = {}  # tid: df
        for i, tid in enumerate(tid_lis):
            df = self.data1_group.get_group(tid).copy()
            tsid = df.tsid.unique().tolist()
            dic = {j: self.tsid_counts[j] for j in tsid}
            user = sorted(dic, key=lambda x: x[1], reverse=True)[:self.tail]

            tsid_lis.extend(tsid)
            user_lis.append(user)
            node_id[tid] = i
            tid_df[tid] = df

        tsid = list(set(tsid_lis))
        ts_vec = self.ts_vec.query(f"tsid in {tsid}")
        tid_len = len(tid_lis)
        for i, j in enumerate(tsid):
            node_id[j] = i + tid_len

        # get node
        user_node = []
        for u in user_lis:
            u_vec = [ts_vec.query(f"tsid == '{i}'").vector.values[0] for i in u]
            user_node.append([sum(x) for x in zip(*u_vec)])
        tsid_node = [ts_vec.query(f"tsid == '{i}'").vector.values[0] for i in tsid]
        node = user_node + tsid_node

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        for tid, df in tid_df.items():
            # user and tsid
            tsid_counts = df.tsid.value_counts().to_dict()
            for tsid, count in tsid_counts.items():
                edge_ind[0].append(node_id[tid])
                edge_ind[1].append(node_id[tsid])
                edge_attr.append(count)
            # tsid and tsid
            df.sort_values(['time'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['idts0'] = df.apply(lambda row: (row.tsid, row.lat, row.lon, row.time), axis=1)
            df['idts1'] = df.groupby('tseg')['idts0'].shift(1)
            spatio = df[~df['idts1'].isna()].copy()  # 获取每个时间片内的空间点，计算空间点距离
            temporal = df[df['idts1'].isna()].copy()  # 获取每个时间片内的第一个点，计算时间片间隔
            temporal['idts1'] = temporal['idts0'].shift(1)
            temporal = temporal[~temporal['idts1'].isna()]

            if spatio.shape[0] > 1:
                spatio['dis'] = spatio.apply(
                    lambda row: (row.idts0[0], row.idts1[0], geodesic(row.idts0[1:3], row.idts1[1:3]).km + 0.1), axis=1)
                for i in spatio.dis.tolist():
                    edge_ind[0].append(node_id[i[0]])  # i[0] tsid0
                    edge_ind[1].append(node_id[i[1]])  # i[1] tsid1
                    edge_attr.append(i[2])

            if temporal.shape[0] > 1:
                temporal['int'] = temporal.apply(
                    lambda row: (row.idts0[0], row.idts1[0], (abs(row.idts0[3] - row.idts1[3]) + 0.1) / 1000), axis=1)
                for i in temporal.int.tolist():
                    edge_ind[0].append(node_id[i[0]])  # i[0] tsid0
                    edge_ind[1].append(node_id[i[1]])  # i[1] tsid1
                    edge_attr.append(i[2])

        return node, edge_ind, edge_attr

    def st_graph(self, tid_lis):
        # get stid 、user stid、 node index
        stid_lis = []
        user_lis = []
        node_id = {}  # tid or stid: ind
        tid_df = {}  # tid: df
        for i, tid in enumerate(tid_lis):
            df = self.data2_group.get_group(tid).copy()
            stid = df.stid.unique().tolist()
            dic = {j: self.stid_counts[j] for j in stid}
            user = sorted(dic, key=lambda x: x[1], reverse=True)[:self.tail]

            stid_lis.extend(stid)
            user_lis.append(user)
            node_id[tid] = i
            tid_df[tid] = df

        stid = list(set(stid_lis))
        st_vec = self.st_vec.query(f"stid in {stid}")
        tid_len = len(tid_lis)
        for i, j in enumerate(stid):
            node_id[j] = i + tid_len

        # get node
        user_node = []
        for u in user_lis:
            u_vec = [st_vec.query(f"stid == '{i}'").vector.values[0] for i in u]
            user_node.append([sum(x) for x in zip(*u_vec)])
        stid_node = [st_vec.query(f"stid == '{i}'").vector.values[0] for i in stid]
        node = user_node + stid_node

        # get edge
        edge_ind = [[], []]
        edge_attr = []
        for tid, df in tid_df.items():
            # user and stid
            stid_counts = df.stid.value_counts().to_dict()
            for stid, count in stid_counts.items():
                edge_ind[0].append(node_id[tid])
                edge_ind[1].append(node_id[stid])
                edge_attr.append(count)
            # stid and stid
            df.sort_values(['time'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            df['idst0'] = df.apply(lambda row: (row.stid, row.lat, row.lon, row.time), axis=1)
            df['idst1'] = df.groupby('sseg')['idst0'].shift(1)
            temporal = df[~df['idst1'].isna()].copy()  # 获取每个空间片内的时间点，计算时间点间隔
            spatio = df[df['idst1'].isna()].copy()  # 获取每个空间片内的第一个点，计算空间片距离
            spatio['idst1'] = spatio['idst0'].shift(1)
            spatio = spatio[~spatio['idst1'].isna()]
            if temporal.shape[0] > 1:
                temporal['inter'] = temporal.apply(
                    lambda row: (row.idst0[0], row.idst1[0], (abs(row.idst0[3] - row.idst1[3]) + 0.1) / 100), axis=1)
                for i in temporal.inter.tolist():
                    edge_ind[0].append(node_id[i[0]])  # i[0] stid0
                    edge_ind[1].append(node_id[i[1]])  # i[1] stid1
                    edge_attr.append(i[2])
            if spatio.shape[0] > 1:
                spatio['dis'] = spatio.apply(
                    lambda row: (row.idst0[0], row.idst1[0], geodesic(row.idst0[1:3], row.idst1[1:3]).km + 0.1), axis=1)
                for i in spatio.dis.tolist():
                    edge_ind[0].append(node_id[i[0]])  # i[0] stid0
                    edge_ind[1].append(node_id[i[1]])  # i[1] stid1
                    edge_attr.append(i[2])

        return node, edge_ind, edge_attr

    def get(self, tid_lis):
        node1, edge_ind1, edge_attr1 = self.ts_graph(tid_lis)
        node2, edge_ind2, edge_attr2 = self.st_graph(tid_lis)

        node1 = torch.tensor(node1, dtype=torch.float32)
        edge_ind1 = torch.tensor(edge_ind1, dtype=torch.long)
        edge_attr1 = torch.tensor(edge_attr1, dtype=torch.float32)
        node2 = torch.tensor(node2, dtype=torch.float32)
        edge_ind2 = torch.tensor(edge_ind2, dtype=torch.long)
        edge_attr2 = torch.tensor(edge_attr2, dtype=torch.float32)

        if 'cuda' in str(self.device):
            node1 = node1.to(device=self.device)
            edge_ind1 = edge_ind1.to(device=self.device)
            edge_attr1 = edge_attr1.to(device=self.device)
            node2 = node2.to(device=self.device)
            edge_ind2 = edge_ind2.to(device=self.device)
            edge_attr2 = edge_attr2.to(device=self.device)

        return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2

    def __getitem__(self, tid):
        return self.get(tid)

