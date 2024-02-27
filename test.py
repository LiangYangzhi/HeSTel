import logging

import numpy as np

from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphLoader import IdDataset, GraphDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

log_path = "./libTrajectory/logs/STEL/"


class Executor(object):
    def __init__(self, tsid_counts, stid_counts):
        self.tsid_counts = tsid_counts
        self.stid_counts = stid_counts
        logging.info(f"Executor...")
        gpu_id = 1
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {self.device}")

    def train(self, train_data, ts_vec, st_vec, out_dim=256, epoch_num=10, batch_size=128, num_workers=0):
        logging.info("train")
        logging.info(f"epoch_num={epoch_num}, batch_size={batch_size}, num_workers={num_workers}")
        data1, data2 = train_data
        index_set = IdDataset(data1)
        graph_data = GraphDataset(data1, ts_vec, self.tsid_counts, data2, st_vec, self.stid_counts, self.device)
        data_loader = DataLoader(index_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # net1
        in_dim1 = len(ts_vec.vector.values[0])
        net1 = GCN(in_dim=in_dim1, out_dim=out_dim).to(self.device)
        lr1 = 0.001
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr1)
        net1.train()
        # net2
        in_dim2 = len(st_vec.vector.values[0])
        net2 = GCN(in_dim=in_dim2, out_dim=out_dim).to(self.device)
        lr2 = 0.001
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr2)
        net2.train()
        # net3
        net3 = GCN(in_dim=out_dim, out_dim=out_dim).to(self.device)
        lr3 = 0.001
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=lr3)
        net3.train()
        logging.info(f"in_dim1={in_dim1}, lr1={lr1}, in_dim2={in_dim2}, lr2={lr2}, lr3={lr3}, out_dim={out_dim}")

        cost = torch.nn.BCEWithLogitsLoss().to(self.device)
        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch + 1}/{epoch_num}')
            batch_loss = 0
            for batch_tid in data_loader:  # 每个批次循环
                node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2 = graph_data[batch_tid]
                # 前向传播
                net1_x = net1(node1, edge_ind1, edge_attr1)
                net2_x = net2(node2, edge_ind2, edge_attr2)
                x1 = net3(net1_x, edge_ind1, edge_attr1)
                x2 = net3(net2_x, edge_ind2, edge_attr2)
                user1 = x1[: batch_size]
                user2 = x2[: batch_size]

                # 计算损失
                labels = torch.arange(len(user1)) == torch.arange(len(user2)).view(-1, 1)
                cosine_sim1 = F.cosine_similarity(user1.unsqueeze(1), user2.unsqueeze(0), dim=2)

                node1_user = node1[: batch_size]
                max_len1 = max(user2.shape[1], node1_user.shape[1])
                node1_user = F.pad(node1_user, (0, max_len1 - node1_user.shape[1]))
                user2 = F.pad(user2, (0, max_len1 - user2.shape[1]))
                cosine_sim2 = F.cosine_similarity(user2.unsqueeze(1), node1_user, dim=2)

                node2_user = node2[: batch_size]
                max_len2 = max(user1.shape[1], node2_user.shape[1])
                node2_user = F.pad(node2_user, (0, max_len2 - node2_user.shape[1]))
                user1 = F.pad(user1, (0, max_len2 - user1.shape[1]))
                cosine_sim3 = F.cosine_similarity(user1.unsqueeze(1), node2_user, dim=2)

                if 'cuda' in str(self.device):
                    labels = labels.to(device=self.device)
                    cosine_sim1 = cosine_sim1.to(device=self.device)
                    cosine_sim2 = cosine_sim2.to(device=self.device)
                    cosine_sim3 = cosine_sim3.to(device=self.device)
                loss1 = cost(cosine_sim1, labels.float())
                loss2 = cost(cosine_sim2, labels.float())
                loss3 = cost(cosine_sim3, labels.float())
                loss = loss1 + loss2 + loss3  # + loss2 + loss3

                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                logging.info(f"batch loss1(A' 与 B' cosine similarity): {loss1.data.item()}")
                logging.info(f"batch loss2(A 与 B' cosine similarity): {loss2.data.item()}")
                logging.info(f"batch loss3(B 与 A' cosine similarity): {loss3.data.item()}")
                logging.info(f"batch loss(loss1 + loss2 +loss3): {loss.data.item()}")
                batch_loss += loss.data.item()

            epoch_loss = batch_loss / len(data_loader)
            logging.info(f"epoch loss: {epoch_loss}")
            torch.save(net1.state_dict(), f'{log_path}net1_parameter-epoch:{epoch}.pth')
            torch.save(net2.state_dict(), f'{log_path}net2_parameter-epoch:{epoch}.pth')
            torch.save(net3.state_dict(), f'{log_path}net3_parameter-epoch:{epoch}.pth')
        torch.save(net1.state_dict(), f'{log_path}net1_parameter.pth')
        torch.save(net2.state_dict(), f'{log_path}net2_parameter.pth')
        torch.save(net3.state_dict(), f'{log_path}net3_parameter.pth')

    def infer(self, test_data, ts_vec, st_vec, in_dim1, out_dim1, in_dim2, out_dim2, parameter1, parameter2):
        logging.info(f"test")

        state_dict1 = torch.load(f'{log_path}{parameter1}')
        net1 = GCN(in_dim=in_dim1, out_dim=out_dim1).to(self.device)
        net1.load_state_dict(state_dict1)
        net1.eval()

        state_dict2 = torch.load(f'{log_path}{parameter2}')
        net2 = GCN(in_dim=in_dim2, out_dim=out_dim2).to(self.device)
        net2.load_state_dict(state_dict2)
        net2.eval()

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

                embedding_1.append(net1(node1, edge_ind1, edge_attr1, batch1).numpy())
                embedding_2.append(net2(node2, edge_ind2, edge_attr2, batch2).numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
