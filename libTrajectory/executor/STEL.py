import logging

import numpy as np

from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphLoader import IdDataset, GraphDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

log_path = "./libTrajectory/logs/STEL/"


class Executor(object):
    def __init__(self):
        logging.info(f"Executor")
        gpu_id = 1
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {self.device}")

    def train(self, train_data, vector, out_dim1=128, out_dim2=128, epoch_num=5, batch_size=64, num_workers=12):
        logging.info(f"train")
        logging.info(f"epoch_num={epoch_num}, batch_size={batch_size}, num_workers={num_workers}")
        data1, data2 = train_data
        ts_vec, st_vec = vector
        index_set = IdDataset(data1)
        graph_data = GraphDataset(data1, ts_vec, data2, st_vec, self.device)
        data_loader = DataLoader(index_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        in_dim1 = len(ts_vec.vector[0])
        model1 = GCN(in_dim=in_dim1, out_dim=out_dim1).to(self.device)
        lr1 = 0.001
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr1)
        model1.train()

        in_dim2 = len(st_vec.vector[0])
        model2 = GCN(in_dim=in_dim2, out_dim=out_dim2).to(self.device)
        lr2 = 0.001
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2)
        model2.train()

        cost = torch.nn.BCEWithLogitsLoss().to(self.device)
        logging.info(
            f"in_dim={in_dim1}, out_dim={out_dim1}, lr1={lr1}, in_dim={in_dim2}, out_dim={out_dim2}, lr2={lr2}")

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch + 1}/{epoch_num}')
            batch_loss = 0
            for batch_tid in data_loader:  # 每个批次循环
                (g1_node, g1_edge_ind, g1_edge_attr, batch1,
                 g2_node, g2_edge_ind, g2_edge_attr, batch2) = graph_data[batch_tid]
                # 前向传播
                g1 = model1(g1_node, g1_edge_ind, g1_edge_attr, batch1)
                g2 = model2(g2_node, g2_edge_ind, g2_edge_attr, batch2)
                # 计算损失
                labels = torch.arange(len(g1)) != torch.arange(len(g2)).view(-1, 1)
                cosine_sim = F.cosine_similarity(g1.unsqueeze(1), g2.unsqueeze(0), dim=2)
                if 'cuda' in str(self.device):
                    labels = labels.to(device=self.device)
                    cosine_sim = cosine_sim.to(device=self.device)
                loss = cost(cosine_sim, labels.float())
                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                logging.info(f"batch loss : {loss.data.item()}")
                batch_loss += loss.data.item()

            epoch_loss = batch_loss / len(data_loader)
            logging.info(f"epoch loss: {epoch_loss}")
            torch.save(model1.state_dict(), f'{log_path}model1_parameter-epoch:{epoch}.pth')
            torch.save(model2.state_dict(), f'{log_path}model2_parameter-epoch:{epoch}.pth')
        torch.save(model1.state_dict(), f'{log_path}model1_parameter.pth')
        torch.save(model2.state_dict(), f'{log_path}model2_parameter.pth')

    def infer(self, test_data, vector, in_dim1, out_dim1, in_dim2, out_dim2, parameter1, parameter2):
        logging.info(f"test")
        ts_vec, st_vec = vector

        state_dict1 = torch.load(f'{log_path}{parameter1}')
        model1 = GCN(in_dim=in_dim1, out_dim=out_dim1).to(self.device)
        model1.load_state_dict(state_dict1)
        model1.eval()

        state_dict2 = torch.load(f'{log_path}{parameter2}')
        model2 = GCN(in_dim=in_dim2, out_dim=out_dim2).to(self.device)
        model2.load_state_dict(state_dict2)
        model2.eval()

        for k, v in test_data.items():
            data1, data2 = v
            graph_data = GraphDataset(data1, ts_vec, data2, st_vec, self.device)
            tid = data1.tid.unique().tolist()
            embedding_1 = []
            embedding_2 = []
            for i in tid:
                node1, edge_ind1, edge_attr1 = graph_data.ts_graph(tid)
                batch1 = []
                for _ in range(len(node1)):
                    batch1.append(i)
                node1 = torch.tensor(node1, dtype=torch.float32)
                edge_ind1 = torch.tensor(edge_ind1, dtype=torch.long)
                edge_attr1 = torch.tensor(edge_attr1)
                batch1 = torch.tensor(batch1, dtype=torch.long)

                node2, edge_ind2, edge_attr2 = graph_data.st_graph(i)
                batch2 = []
                for _ in range(len(node2)):
                    batch2.append(i)
                node2 = torch.tensor(node2, dtype=torch.float32)
                edge_ind2 = torch.tensor(edge_ind2, dtype=torch.long)
                edge_attr2 = torch.tensor(edge_attr2)
                batch2 = torch.tensor(batch2, dtype=torch.long)

                if 'cuda' in str(self.device):
                    node1 = node1.to(device=self.device)
                    edge_ind1 = edge_ind1.to(device=self.device)
                    edge_attr1 = edge_attr1.to(device=self.device)
                    batch1 = torch.tensor(batch1, dtype=torch.long)
                    node2 = node2.to(device=self.device)
                    edge_ind2 = edge_ind2.to(device=self.device)
                    edge_attr2 = edge_attr2.to(device=self.device)
                    batch2 = torch.tensor(batch2, dtype=torch.long)

                embedding_1.append(model1(node1, edge_ind1, edge_attr1, batch1).numpy())
                embedding_2.append(model2(node2, edge_ind2, edge_attr2, batch2).numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
