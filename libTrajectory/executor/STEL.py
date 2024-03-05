import logging
import time

import numpy as np
from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphLoader import GraphDataset, IdDataset
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

log_path = "./libTrajectory/logs/STEL/"


class Executor(object):
    def __init__(self, st_vec, stid_counts, batch_size=128, num_workers=3):
        self.st_vec = st_vec
        self.stid_counts = stid_counts
        self.batch_size = batch_size
        self.num_workers = num_workers
        logging.info(f"Executor...")
        gpu_id = 1
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"device={self.device}, batch_size={self.batch_size}, num_workers={self.num_workers}")

    def train(self, train_data, mid_dim=128, out_dim=256, epoch_num=3):
        logging.info(f"train, epoch_num={epoch_num}")
        data1, data2 = train_data
        index_set = IdDataset(data1)
        data_loader = DataLoader(dataset=index_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        graph_data = GraphDataset(data1, data2, self.st_vec, self.stid_counts, train=True)

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

        cost = torch.nn.CrossEntropyLoss().to(self.device)
        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch + 1}/{epoch_num}')
            batch_loss = 0
            for data in data_loader:  # 每个批次循环
                node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2, struct = graph_data[data]
                logging.info('batch')
                if 'cuda' in str(self.device):
                    node1 = node1.to(device=self.device)
                    edge_ind1 = edge_ind1.to(device=self.device)
                    edge_attr1 = edge_attr1.to(device=self.device)
                    node2 = node2.to(device=self.device)
                    edge_ind2 = edge_ind2.to(device=self.device)
                    edge_attr2 = edge_attr2.to(device=self.device)

                # 前向传播
                net1_x = net1(node1, edge_ind1, edge_attr1)
                net2_x = net2(node2, edge_ind2, edge_attr2)
                x1 = net3(net1_x, edge_ind1, edge_attr1)
                x2 = net3(net2_x, edge_ind2, edge_attr2)

                # 计算损失
                # 增强样本1 loss1  "enh1": [[A A'], ...]
                label = []
                vector1 = []
                for i, pairs in enumerate(struct['enh1']):
                    vector1.append(x1[pairs[0]])
                    label.append(i)
                    vector1.append(x1[pairs[1]])
                    label.append(i)
                label = torch.tensor(label)
                vector = torch.stack(vector1, dim=0)
                if 'cuda' in str(self.device):
                    label = label.to(device=self.device)
                    vector = vector.to(device=self.device)
                loss1 = cost(vector, label)

                # 增强样本2 loss2  "enh1": [[B B'], ...]
                label = []
                vector = []
                for i, pairs in enumerate(struct['enh1']):
                    vector.append(x2[pairs[0]])
                    label.append(i)
                    vector.append(x2[pairs[1]])
                    label.append(i)
                label = torch.tensor(label)
                vector = torch.stack(vector, dim=0)
                if 'cuda' in str(self.device):
                    label = label.to(device=self.device)
                    vector = vector.to(device=self.device)
                loss2 = cost(vector, label)

                # 正样本对 loss3  "ps": [[A B], ...]、负样本对1 loss  "ns1": [[A B], ...]  "ns2": [[B A], ...]
                label = []
                vector1 = []
                vector2 = []
                for pairs in struct['ps']:  # [A B]
                    label.append(1)
                    vector1.append(x1[pairs[0]])
                    vector2.append(x2[pairs[1]])

                for pairs in struct['ns1']:  # [A B]
                    label.append(0)
                    vector1.append(x1[pairs[0]])
                    vector2.append(x2[pairs[1]])

                for pairs in struct['ns2']:  # [B A]
                    label.append(0)
                    vector1.append(x1[pairs[1]])
                    vector2.append(x2[pairs[0]])

                label = torch.tensor(label, dtype=torch.float32)
                vector1 = torch.stack(vector1, dim=0)
                vector2 = torch.stack(vector2, dim=0)
                if 'cuda' in str(self.device):
                    label = label.to(device=self.device)
                    vector1 = vector1.to(device=self.device)
                    vector2 = vector2.to(device=self.device)
                similarities = F.cosine_similarity(vector1, vector2)
                loss3 = cost(similarities, label)

                loss = loss1 + loss2 + loss3
                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                logging.info(f"batch loss(loss1={loss1} + loss2={loss2} + loss3={loss3}): {loss.data.item()}")
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
            index_set = IdDataset(data1)
            graph_data = GraphDataset(data1, data2, self.st_vec, self.stid_counts, train=False)
            data_loader = DataLoader(index_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            embedding_1 = []
            embedding_2 = []
            for batch_tid in data_loader:
                node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2 = graph_data[batch_tid]
                if 'cuda' in str(self.device):
                    node1 = node1.to(device=self.device)
                    edge_ind1 = edge_ind1.to(device=self.device)
                    edge_attr1 = edge_attr1.to(device=self.device)
                    node2 = node2.to(device=self.device)
                    edge_ind2 = edge_ind2.to(device=self.device)
                    edge_attr2 = edge_attr2.to(device=self.device)
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
