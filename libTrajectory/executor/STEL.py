import logging
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from libTrajectory.model.GraphTransformer import GraphTransformer
from libTrajectory.model.loss import dis_loss, infoNCE_loss, cosine_loss
from libTrajectory.preprocessing.STEL.batch_collate import train_coll, infer_coll
from libTrajectory.preprocessing.STEL.graphDataset import GraphLoader
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader


class Executor(object):
    def __init__(self, path, log_path, in_dim=34, out_dim=32, cuda=1, net_name='net'):
        self.path = path
        self.log_path = log_path
        logging.info(f"Executor...")
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net_name = net_name
        logging.info(f"device={self.device}, in_dim={self.in_dim}, out_dim={self.out_dim}")

    def train(self, train_tid, enhance_ns, test_tid=None, epoch_num=1, batch_size=64, num_workers=16):
        logging.info(f"train, epoch_num={epoch_num}, batch_size={batch_size}, num_workers={num_workers}")
        graph_data = GraphLoader(self.path, train_tid, train=True, enhance_ns=enhance_ns)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, drop_last=True, shuffle=True)
        # net
        self.heads = 4
        net = GraphTransformer(self.in_dim, self.out_dim, self.heads).to(self.device)
        lr = 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)

        logging.info(f"lr={lr}, head={self.heads}")
        penalty1 = 0.1
        logging.info(f"loss1: sim_loss penalty = {penalty1}")

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_loss1 = []
            epoch_loss2 = []
            epoch_loss3 = []
            epoch_loss4 = []
            for (node, edge, edge_attr, global_spatial, global_temporal,
                 tid1, tid2, ps1, ps2, ns1, ns2) in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(self.device):
                    node = node.to(device=self.device)
                    edge = edge.to(device=self.device)
                    edge_attr = edge_attr.to(device=self.device)
                    global_spatial = global_spatial.to(device=self.device)
                    global_temporal = global_temporal.to(device=self.device)

                # 前向传播
                x = net(node, edge, edge_attr, global_spatial, global_temporal)
                tid1_vec = x[tid1].to(device=self.device)
                tid2_vec = x[tid2].to(device=self.device)
                ps1_vec = x[ps1].to(device=self.device)
                ps2_vec = x[ps2].to(device=self.device)

                # A -- B random sample
                # sim = torch.matmul(tid1_vec, tid2_vec.T)
                # loss1 = dis_loss(sim, penalty=penalty1)
                # epoch_loss1.append(loss1.data.item())

                # A --- enhance negative B
                sim1 = []
                for ind, i in enumerate(ns2):
                    ns2_vec = x[i]
                    sim1.append(torch.matmul(tid1_vec[ind], ns2_vec.T))
                sim1 = torch.stack(sim1)
                loss1 = dis_loss(sim1, penalty=penalty1)
                epoch_loss1.append(loss1.data.item())

                # B --- enhance negative A
                sim2 = []
                for ind, i in enumerate(ns1):
                    ns1_vec = x[i]
                    sim2.append(torch.matmul(tid2_vec[ind], ns1_vec.T))
                sim2 = torch.stack(sim2)
                loss2 = dis_loss(sim2, penalty=penalty1)
                epoch_loss2.append(loss2.data.item())

                # A --- enhance positive A
                sim3 = F.cosine_similarity(tid1_vec.unsqueeze(1), ps1_vec.unsqueeze(1), dim=2)
                loss3 = dis_loss(sim3)
                epoch_loss3.append(loss3.data.item())

                # loss3 = infoNCE_loss(tid1_vec, ps1_vec)
                # epoch_loss3.append(loss3.data.item())

                # sim3 = torch.matmul(tid1_vec, ps1_vec.T)
                # loss3 = dis_loss(sim3)
                # epoch_loss3.append(loss3.data.item())

                # vec = torch.cat((tid1_vec, ps1_vec), dim=0)
                # label = torch.tensor([i for i in range(len(tid1))] * 2).to(device=self.device)
                # loss3 = cross_entropy(vec, label)
                # epoch_loss3.append(loss3.data.item())

                # sim = F.cosine_similarity(tid1_vec.unsqueeze(1), ps1_vec.unsqueeze(1), dim=2)
                # loss3 = cosine_loss(sim)
                # epoch_loss3.append(loss3.data.item())

                # B --- enhance positive B
                sim4 = F.cosine_similarity(tid2_vec.unsqueeze(1), ps2_vec.unsqueeze(1), dim=2)
                loss4 = dis_loss(sim4)
                epoch_loss4.append(loss4.data.item())

                # loss4 = infoNCE_loss(tid2_vec, ps2_vec)
                # epoch_loss4.append(loss4.data.item())

                # sim4 = torch.matmul(tid2_vec, ps2_vec.T)
                # loss4 = dis_loss(sim4)
                # epoch_loss4.append(loss4.data.item())

                # vec = torch.cat((tid2_vec, ps2_vec), dim=0)
                # label = torch.tensor([i for i in range(len(tid2))] * 2).to(device=self.device)
                # loss4 = cross_entropy(vec, label)
                # epoch_loss4.append(loss4.data.item())

                # sim = F.cosine_similarity(tid2_vec.unsqueeze(1), ps2_vec.unsqueeze(1), dim=2)
                # loss4 = cosine_loss(sim)
                # epoch_loss4.append(loss4.data.item())

                # loss = loss1
                loss = loss1 + loss2 + 2*loss3 + 2*loss4  #
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # logging.info(f"loss1:{loss1}")
                logging.info(f"loss1:{loss1} + loss2:{loss2} + loss3:{loss3} + loss4:{loss4} = {loss.data.item()}")

            epoch_loss = epoch_loss / len(data_loader)
            epoch_loss1 = sorted(epoch_loss1)
            epoch_loss2 = sorted(epoch_loss2)
            epoch_loss3 = sorted(epoch_loss3)
            epoch_loss4 = sorted(epoch_loss4)
            logging.info(f"epoch loss:{epoch_loss}")
            logging.info(f"epoch loss1 min:{epoch_loss1[:2]}, max:{epoch_loss1[-1]}, mean:{sum(epoch_loss1)/len(epoch_loss1)}")
            logging.info(f"epoch loss2 min:{epoch_loss2[:2]}, max:{epoch_loss2[-1]}, mean:{sum(epoch_loss2)/len(epoch_loss2)}")
            logging.info(f"epoch loss3 min:{epoch_loss3[:2]}, max:{epoch_loss3[-1]}, mean:{sum(epoch_loss3)/len(epoch_loss3)}")
            logging.info(f"epoch loss4 min:{epoch_loss4[:2]}, max:{epoch_loss4[-1]}, mean:{sum(epoch_loss4)/len(epoch_loss4)}")

            print(f"epoch loss1 min:{epoch_loss1[:2]}, max:{epoch_loss1[-1]}, mean:{sum(epoch_loss1)/len(epoch_loss1)}")
            print(f"epoch loss2 min:{epoch_loss2[:2]}, max:{epoch_loss2[-1]}, mean:{sum(epoch_loss2)/len(epoch_loss2)}")
            print(f"epoch loss3 min:{epoch_loss3[:2]}, max:{epoch_loss3[-1]}, mean:{sum(epoch_loss3)/len(epoch_loss3)}")
            print(f"epoch loss4 min:{epoch_loss4[:2]}, max:{epoch_loss4[-1]}, mean:{sum(epoch_loss4)/len(epoch_loss4)}")
            net_name = f"{self.net_name}{epoch}"
            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            if test_tid is not None:
                self.infer(test_tid, net_name=net_name)
        torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')

    def infer(self, test_data, net_name=None):
        logging.info("test")
        if net_name is None:
            net_name = self.net_name
        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth')
        # net = GCN(self.in_dim, self.out_dim).to(self.device)
        net = GraphTransformer(self.in_dim, self.out_dim, self.heads).to(self.device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={self.net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=16, num_workers=12, collate_fn=infer_coll,
                                     persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(self.device):
                    node = node.to(device=self.device)
                    edge = edge.to(device=self.device)
                    edge_attr = edge_attr.to(device=self.device)
                    global_spatial = global_spatial.to(device=self.device)
                    global_temporal = global_temporal.to(device=self.device)
                x = net(node, edge, edge_attr, global_spatial, global_temporal)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(self.device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)
