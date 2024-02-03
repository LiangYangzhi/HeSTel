from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphLoader import IdDataset, GraphDataset
from torch.utils.data import DataLoader
import torch


class Executor(object):

    def train(self, data1, ts_vec, data2, st_vec, out_dim=128, epoch_num=10, batch_size=32):
        index_set = IdDataset(data1)
        graph_data = GraphDataset(data1, ts_vec, data2, st_vec)
        in_dim1 = len(ts_vec.vector[0])
        in_dim2 = len(st_vec.vector[0])
        out_dim = 64
        data_loader = DataLoader(index_set, batch_size=batch_size, shuffle=True)

        model1 = GCN(in_dim=in_dim1, out_dim=out_dim)
        model2 = GCN(in_dim=in_dim2, out_dim=out_dim)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        model1.train()
        model2.train()

        for epoch in range(epoch_num):  # 每个epoch循环
            print(f'Epoch {epoch + 1}/{epoch_num}')
            for batch_tid in data_loader:  # 每个批次循环
                (g1_node, g1_edge_ind, g1_edge_attr, batch1,
                 g2_node, g2_edge_ind, g2_edge_attr, batch2) = graph_data[batch_tid]
                # 前向传播
                g1 = model1(g1_node, g1_edge_ind, g1_edge_attr, batch1)
                g2 = model2(g2_node, g2_edge_ind, g2_edge_attr, batch2)
                labels = torch.arange(len(g1)) != torch.arange(len(g2)).view(-1, 1)
                # 计算损失
                cosine_sim = F.cosine_similarity(g1.unsqueeze(1), g2.unsqueeze(0), dim=2)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(cosine_sim, labels.float())
                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            print(f'Epoch {epoch + 1}: Loss = {loss.item()}')

    def infer(self):
        pass
