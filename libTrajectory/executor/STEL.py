import logging

import numpy as np

from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphLoader import IdDataset, GraphDataset
from libTrajectory.evaluator.faiss_cosine import evaluator
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

log_path = "./libTrajectory/logs/STEL/"


class Executor(object):
    def __init__(self, st_vec, stid_counts, batch_size=256, num_workers=2):
        self.st_vec = st_vec
        self.stid_counts = stid_counts
        self.batch_size = batch_size
        self.num_workers = num_workers
        logging.info(f"Executor...")
        gpu_id = 1
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"device={self.device}, batch_size={self.batch_size}, num_workers={self.num_workers}")

    def train(self, train_data, mid_dim=256, out_dim=512, epoch_num=3):
        logging.info("train")
        logging.info(f"epoch_num={epoch_num}")
        data1, data2 = train_data
        index_set = IdDataset(data1)
        graph_data = GraphDataset(data1, data2, self.st_vec, self.stid_counts, self.device)
        data_loader = DataLoader(index_set, batch_size=self.batch_size, shuffle=True, persistent_workers=True, num_workers=self.num_workers)

        in_dim = len(self.st_vec.vec.values[0])
        # net1
        net1 = GCN(in_dim=in_dim, out_dim=mid_dim, device=self.device).to(self.device)
        lr1 = 0.001
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr1)
        net1.train()
        # net2
        net2 = GCN(in_dim=in_dim, out_dim=mid_dim, device=self.device).to(self.device)
        lr2 = 0.001
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr2)
        net2.train()
        # net3
        net3 = GCN(in_dim=mid_dim, out_dim=out_dim, device=self.device).to(self.device)
        lr3 = 0.001
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=lr3)
        net3.train()
        logging.info(f"in_dim={in_dim}, mid_dim={mid_dim}, out_dim={out_dim}, lr1={lr1}, lr2={lr2}, lr3={lr3}")

        cost = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.batch_size / 2])).to(self.device)
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
                user1 = x1[: len(batch_tid)]
                user2 = x2[: len(batch_tid)]

                # 计算损失
                labels = torch.arange(len(user1)) == torch.arange(len(user2)).view(-1, 1)
                cosine_sim1 = F.cosine_similarity(user1.unsqueeze(1), user2, dim=2)

                node1_user = node1[: len(batch_tid)]
                max_len1 = max(user2.shape[1], node1_user.shape[1])
                node1_user = F.pad(node1_user, (0, max_len1 - node1_user.shape[1]))
                user2 = F.pad(user2, (0, max_len1 - user2.shape[1]))
                cosine_sim2 = F.cosine_similarity(user2.unsqueeze(1), node1_user, dim=2)

                node2_user = node2[: len(batch_tid)]
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

    def infer(self, test_data, mid_dim=256, out_dim=512,
              para1='net1_parameter-epoch:1.pth', para2='net2_parameter-epoch:1.pth', para3='net3_parameter-epoch:1.pth'):
        logging.info("test")
        in_dim = len(self.st_vec.vec.values[0])

        state_dict1 = torch.load(f'{log_path}{para1}')
        net1 = GCN(in_dim=in_dim, out_dim=mid_dim, device=self.device).to(self.device)
        net1.load_state_dict(state_dict1)
        net1.eval()

        state_dict2 = torch.load(f'{log_path}{para2}')
        net2 = GCN(in_dim=in_dim, out_dim=mid_dim, device=self.device).to(self.device)
        net2.load_state_dict(state_dict2)
        net2.eval()

        state_dict3 = torch.load(f'{log_path}{para3}')
        net3 = GCN(in_dim=mid_dim, out_dim=out_dim, device=self.device).to(self.device)
        net3.load_state_dict(state_dict3)
        net3.eval()
        logging.info(f"in_dim={in_dim}, mid_dim={mid_dim}, out_dim={out_dim}, net1={para1}, net2={para2}, net3={para3}")

        for k, v in test_data.items():
            logging.info(f"{k}...")
            data1, data2 = v
            # test_tid = data1.tid[:20].tolist()
            # data1 = data1.query(f"tid in {test_tid}")
            # data2 = data2.query(f"tid in {test_tid}")
            index_set = IdDataset(data1)
            graph_data = GraphDataset(data1, data2, self.st_vec, self.stid_counts, self.device)
            data_loader = DataLoader(index_set, batch_size=self.batch_size, shuffle=False, persistent_workers=True, num_workers=self.num_workers)
            embedding_1 = []
            embedding_2 = []
            for batch_tid in data_loader:
                node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2 = graph_data[batch_tid]
                net1_x = net1(node1, edge_ind1, edge_attr1)
                net2_x = net2(node2, edge_ind2, edge_attr2)
                x1 = net3(net1_x, edge_ind1, edge_attr1)
                x2 = net3(net2_x, edge_ind2, edge_attr2)
                if 'cuda' in str(self.device):
                    x1 = x1.to(device='cpu')
                    x2 = x2.to(device='cpu')
                for i in range(len(batch_tid)):
                    embedding_1.append(x1[i].detach().numpy())
                    embedding_2.append(x2[i].detach().numpy())
                embedding_1 = np.array(embedding_1)
                embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)
