import logging
import random
import torch
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.model.STEL import DualTowerGCN


class PairGraphData(DataLoader):
    def __init__(self, tid1, graph1, tid2, graph2):
        self.tid1 = tid1
        self.graph1 = graph1
        self.tid2 = tid2
        self.graph2 = graph2
        self.pairs_p, self.pairs_n = self.create_pairs()

    def __len__(self):
        return len(self.pairs_p)

    def create_pairs(self):
        pairs_p = []  # positive pairs
        pairs_n = []  # negative pairs
        for id1, t1 in enumerate(self.tid1):
            id2 = self.tid2.index(t1)
            pairs_p.append((self.graph1[id1], self.graph2[id2]))

            while True:
                i = random.randint(0, len(self.tid2) - 1)
                if i != id2:
                    pairs_n.append((self.graph1[id1], self.graph2[i]))
                    break
        return pairs_p, pairs_n

    def get(self, idx):
        pairs_p = self.pairs_p[idx]
        pairs_n = self.pairs_n[idx]
        positive_1, positive_0 = self.graph1[pairs_p[0]], self.graph1[pairs_p[1]]
        negative_1, negative_0 = self.graph2[pairs_n[0]], self.graph2[pairs_n[1]]
        return positive_1, positive_0, negative_1, negative_0

    def __getitem__(self, idx):
        return self.get(idx)


if __name__ == "__main__":
    preprocessor = Preprocessor("./libTrajectory/dataset/AIS/test10.csv")
    # graph: [[torch.tensor(node), torch.tensor(edge_ind), torch.tensor(edge_attr)], ]
    # tid : 用户标识，tid1与tid2相同则为正样本，否则为负样本
    tid1, graph1, tid2, graph2 = preprocessor.get()
    print(f"tid1 len={len(tid1)}, tid2 len={len(tid2)}, graph1 len={len(graph1)}, graph2 len={len(graph2)}")
    print(f"node1 len={graph1[0][0].shape}, node2 len={graph2[0][0].shape}")
    dataset = PairGraphData(tid1, graph1, tid2, graph2)
    # train_loader = DataLoader(dataset, batch_size=1)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DualTowerGCN(in_dim1=graph1[0][0].shape[1], out_dim1=64, in_dim2=graph2[0][0].shape[1], out_dim2=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    epoch_num = 10
    for epoch in range(epoch_num):  # 每个epoch循环
        for positive_1, positive_0, negative_1, negative_0 in train_loader:   # 每个批次循环
            logging.info(f'Epoch {epoch+1}/{epoch_num}')
            # 前向传播
            pos_sim = model(positive_1[0], positive_1[1], positive_1[2],
                            positive_0[0], positive_0[1], positive_0[2])
            neg_sim = model(negative_1[0], negative_1[1], negative_1[2],
                            negative_0[0], negative_0[1], negative_0[2])
            # 计算损失
            loss = (F.binary_cross_entropy(pos_sim, torch.ones_like(pos_sim)) +
                    F.binary_cross_entropy(neg_sim, torch.zeros_like(neg_sim)))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}: Loss = {loss.item()}')
