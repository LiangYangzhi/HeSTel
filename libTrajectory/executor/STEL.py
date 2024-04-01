import logging
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from libTrajectory.model.GCN import GCN
from libTrajectory.model.GraphTransformer import GraphTransformer
from libTrajectory.model.loss import sim_loss
from libTrajectory.preprocessing.STEL.batch_collate import collate_fun
from libTrajectory.preprocessing.STEL.graphDataset import GraphLoader
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader


class Executor(object):
    def __init__(self, path, stid_counts, log_path, in_dim=32, mid_dim=128, out_dim=128, cuda=0):
        self.path = path
        self.log_path = log_path
        self.stid_counts = stid_counts
        logging.info(f"Executor...")
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        logging.info(f"device={self.device}, in_dim={self.in_dim}, mid_dim={self.mid_dim}, out_dim={self.out_dim}")

    def train(self, train_tid, enhance_ns, test_tid=None, epoch_num=1, batch_size=64, num_workers=16):
        logging.info(f"train, epoch_num={epoch_num}, batch_size={batch_size}, num_workers={num_workers}")
        graph_data = GraphLoader(self.path, train_tid, self.stid_counts, train=True, enhance_ns=enhance_ns)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=collate_fun, persistent_workers=True, drop_last=True, shuffle=True)
        # net1
        net1 = GCN(self.in_dim, self.mid_dim, self.device).to(self.device)
        lr1 = 0.002
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr1)
        net1.train()
        # net2
        net2 = GCN(self.in_dim, self.mid_dim, self.device).to(self.device)
        lr2 = 0.002
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr2)
        net2.train()
        # net3
        net3 = GCN(self.mid_dim, self.out_dim, self.device).to(self.device)
        lr3 = 0.002
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=lr3)
        net3.train()
        logging.info(f"lr1={lr1}, lr2={lr2}, lr3={lr3}")
        penalty1 = 0.1
        logging.info(f"loss1: sim_loss penalty = {penalty1}")

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_loss1 = []
            # epoch_loss2 = []
            # epoch_loss3 = []
            for (node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2,
                 tid1, tid2, ps1, ps2, ns1, ns2) in tqdm(data_loader):  # 每个批次循环
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

                tid1_vec = x1[tid1].to(device=self.device)
                tid2_vec = x2[tid2].to(device=self.device)
                # ps1_vec = x1[ps1].to(device=self.device)
                # ps2_vec = x2[ps2].to(device=self.device)
                # ns1_vec = x1[ns1].to(device=self.device)
                # ns2_vec = x2[ns2].to(device=self.device)

                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss1 = sim_loss(sim1, penalty=penalty1)
                epoch_loss1.append(loss1.data.item())

                loss = loss1
                # loss.requires_grad_(True)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                # logging.info(f"loss1:{loss1} + loss2:{loss2} + loss3:{loss3} = {loss.data.item()}")
                logging.info(f"loss1:{loss1}")

            epoch_loss = epoch_loss / len(data_loader)
            epoch_loss1 = sorted(epoch_loss1)
            # epoch_loss2 = sorted(epoch_loss2)
            # epoch_loss3 = sorted(epoch_loss3)
            logging.info(f"epoch loss:{epoch_loss}")
            logging.info(f"epoch loss1 min:{epoch_loss1[:2]}, max:{epoch_loss1[-1]}, mean:{sum(epoch_loss1)/len(epoch_loss1)}")
            # logging.info(f"epoch loss2 min:{epoch_loss2[:2]}, max:{epoch_loss2[-1]}, mean:{sum(epoch_loss2)/len(epoch_loss2)}")
            # logging.info(f"epoch loss3 min:{epoch_loss3[:2]}, max:{epoch_loss3[-1]}, mean:{sum(epoch_loss3)/len(epoch_loss3)}")

            print(f"epoch loss1 min:{epoch_loss1[:2]}, max:{epoch_loss1[-1]}, mean:{sum(epoch_loss1)/len(epoch_loss1)}")
            # print(f"epoch loss2 min:{epoch_loss2[:2]}, max:{epoch_loss2[-1]}, mean:{sum(epoch_loss2)/len(epoch_loss2)}")
            # print(f"epoch loss3 min:{epoch_loss3[:2]}, max:{epoch_loss3[-1]}, mean:{sum(epoch_loss3)/len(epoch_loss3)}")
            torch.save(net1.state_dict(), f'{self.log_path}net1_parameter-epoch:{epoch}.pth')
            torch.save(net2.state_dict(), f'{self.log_path}net2_parameter-epoch:{epoch}.pth')
            torch.save(net3.state_dict(), f'{self.log_path}net3_parameter-epoch:{epoch}.pth')
            if test_tid is not None:
                self.infer(test_tid, para1=f'net1_parameter-epoch:{epoch}.pth',
                           para2=f'net2_parameter-epoch:{epoch}.pth', para3=f'net3_parameter-epoch:{epoch}.pth')

    def infer(self, test_data, para1='net1_parameter-epoch:0.pth',
              para2='net2_parameter-epoch:0.pth', para3='net3_parameter-epoch:0.pth'):
        logging.info("test")
        state_dict1 = torch.load(f'{self.log_path}{para1}')
        net1 = GCN(self.in_dim, self.mid_dim, self.device).to(self.device)
        net1.load_state_dict(state_dict1)
        net1.eval()

        state_dict2 = torch.load(f'{self.log_path}{para2}')
        net2 = GCN(self.in_dim, self.mid_dim, self.device).to(self.device)
        net2.load_state_dict(state_dict2)
        net2.eval()

        state_dict3 = torch.load(f'{self.log_path}{para3}')
        net3 = GCN(self.mid_dim, self.out_dim, self.device).to(self.device)
        net3.load_state_dict(state_dict3)
        net3.eval()
        logging.info(f"net1={para1}, net2={para2}, net3={para3}")

        for k, v in test_data.items():
            logging.info(f"{k}...")
            graph_data = GraphLoader(self.path, v, self.stid_counts, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=8, num_workers=12, collate_fn=lambda x: x,
                                     persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for struct in data_loader:
                for data in struct:
                    node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2 = data

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
                    net1_x = net1(node1, edge_ind1, edge_attr1)
                    net2_x = net2(node2, edge_ind2, edge_attr2)
                    x1 = net3(net1_x, edge_ind1, edge_attr1)
                    x2 = net3(net2_x, edge_ind2, edge_attr2)
                    if 'cuda' in str(self.device):
                        x1 = x1.to(device='cpu')
                        x2 = x2.to(device='cpu')
                    embedding_1.append(x1[0].detach().numpy())
                    embedding_2.append(x2[0].detach().numpy())

            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

