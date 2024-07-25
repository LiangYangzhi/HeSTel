import logging
import pickle

import numpy as np
from tqdm import tqdm

from libTrajectory.model.hst import HST
from libTrajectory.model.ab_hst_net import GCN, GTransformer, GAT
from libTrajectory.model.loss import dis_loss, infoNCE_loss
from libTrajectory.preprocessing.STEL.batch_collate import train_coll, infer_coll
from libTrajectory.preprocessing.STEL.graphDataset import GraphLoader
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader


class Executor(object):
    def __init__(self, path, log_path, config):
        logging.info(f"Executor...")
        logging.info(f"config: {config}")
        self.path = path
        self.log_path = log_path
        self.config = config
        cuda = self.config['cuda']
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.in_dim = self.config['in_dim']
        self.out_dim = self.config['out_dim']
        self.net_name = self.config['net_name']
        logging.info(f"device={self.device}, in_dim={self.in_dim}, out_dim={self.out_dim}")

    def train(self, train_tid, enhance_tid, test_tid=None):
        if self.config.get("loss") == "crossEntropy":
            self.ab_cross_entropy_loss(train_tid, enhance_tid, test_tid=test_tid)
            return None
        elif self.config.get("loss") == "infoNCE":
            self.ab_infoNCE_loss(train_tid, enhance_tid, test_tid=test_tid)
            return None
        elif self.config.get("loss") == "tripletMargin":
            self.ab_triplet_margin_loss(train_tid, enhance_tid, test_tid=test_tid)
            return None

        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        self.heads = 4
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        if self.config.get("net") == "GCN":
            net = GCN(self.in_dim, self.out_dim*self.heads).to(self.device)
        elif self.config.get("net") == "GAT":
            net = GAT(self.in_dim, self.out_dim, self.heads).to(self.device)
        elif self.config.get("net") == "GTransformer":
            net = GTransformer(self.in_dim, self.out_dim, self.heads).to(self.device)
        lr = self.config['lr']  # 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()

        logging.info(f"lr={lr}, head={self.heads}")
        penalty = self.config['penalty']  # 0.1
        logging.info(f"full_loss: dis_loss penalty = {penalty}")
        tids = []
        ratio_full = self.config['ratio_full']
        ratio_ns2 = self.config['ratio_ns2'] if ns_num else 0
        ratio_ns1 = self.config['ratio_ns1'] if ns_num else 0
        ratio_ps1 = self.config['ratio_ps1'] if ps_num else 0
        ratio_ps2 = self.config['ratio_ps2'] if ps_num else 0

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = [], [], [], [], []
            for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2, batch_tid in tqdm(data_loader):  # 每个批次循环
                tids += batch_tid
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

                # full_loss: A -- B random sample
                full_loss = 0
                if ratio_full:
                    sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                    full_loss = dis_loss(sim1, penalty=penalty)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    sim2 = []
                    for ind, i in enumerate(ns2):
                        ns2_vec = x[i]
                        sim2.append(torch.matmul(tid1_vec[ind], ns2_vec.T))
                    sim2 = torch.stack(sim2)
                    ns2_loss = dis_loss(sim2, penalty=penalty)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    sim3 = []
                    for ind, i in enumerate(ns1):
                        ns1_vec = x[i]
                        sim3.append(torch.matmul(tid2_vec[ind], ns1_vec.T))
                    sim3 = torch.stack(sim3)
                    ns1_loss = dis_loss(sim3, penalty=penalty)
                    epoch_ns1.append(ns1_loss.data.item())

                # ps1_loss: enhance positive A --- enhance negative B
                ps1_loss = 0
                if ratio_ps1:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps1]
                        ps1_col_vec = x[columns]
                        sim4 = []
                        for ind, j in enumerate(ns2):
                            ns2_vec = x[j]
                            sim4.append(torch.matmul(ps1_col_vec[ind], ns2_vec.T))  # ns2_vec   tid2_vec
                        sim4 = torch.stack(sim4)
                        ps1_loss += dis_loss(sim4, penalty=penalty)
                    epoch_ps1.append(ps1_loss.data.item())

                # loo5: enhance positive B --- enhance negative A
                ps2_loss = 0
                if ratio_ps2:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps2]
                        ps2_col_vec = x[columns]
                        sim5 = []
                        for ind, j in enumerate(ns2):
                            ns1_vec = x[j]
                            sim5.append(torch.matmul(ps2_col_vec[ind], ns1_vec.T))  # ns1_vec   tid1_vec
                        sim5 = torch.stack(sim5)
                        ps2_loss += dis_loss(sim5, penalty=penalty)
                    epoch_ps2.append(ps2_loss.data.item())

                # loss = 50*full_loss + 10*ns2_loss + 10*ns1_loss + ps1_loss + ps2_loss
                loss = ratio_full * full_loss + ratio_ns2 * ns2_loss + ratio_ns1 * ns1_loss + ratio_ps1 * ps1_loss + ratio_ps2 * ps2_loss
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f"{full_loss} + {ns2_loss} + {ns1_loss} + {ps1_loss} + {ps2_loss} = {loss.data.item()}")
            torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')

            epoch_loss = epoch_loss / len(data_loader)
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = sorted(epoch_full), sorted(epoch_ns2), sorted(epoch_ns1), sorted(epoch_ps1), sorted(epoch_ps2)
            logging.info(f"epoch loss:{epoch_loss}")
            if ratio_full:
                mean_full = sum(epoch_full)/len(epoch_full)
                logging.info(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")
                print(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")

            if ratio_ns1:
                mean_ns1 = sum(epoch_ns1)/len(epoch_ns1)
                logging.info(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
                print(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
            if ratio_ns2:
                mean_ns2 = sum(epoch_ns2)/len(epoch_ns2)
                logging.info(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
                print(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
            if ratio_ps1:
                mean_ps1 = sum(epoch_ps1)/len(epoch_ps1)
                logging.info(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
                print(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
            if ratio_ps2:
                mean_ps2 = sum(epoch_ps2)/len(epoch_ps2)
                logging.info(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")
                print(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")

            if test_tid is not None:
                self.infer(test_tid, net_name=self.net_name)

    def infer(self, test_data, net_name=None):
        logging.info("test")
        if net_name is None:
            net_name = self.net_name
        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth')
        # net = GCN(self.in_dim, self.out_dim).to(self.device)
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        if self.config.get("net") == "GCN":
            net = GCN(self.in_dim, self.out_dim*self.heads).to(self.device)
        elif self.config.get("net") == "GAT":
            net = GAT(self.in_dim, self.out_dim, self.heads).to(self.device)
        elif self.config.get("net") == "GTransformer":
            net = GTransformer(self.in_dim, self.out_dim, self.heads).to(self.device)
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

    def ab_cross_entropy_loss(self, train_tid, enhance_tid, test_tid=None):
        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        self.heads = 4
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        lr = self.config['lr']  # 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()

        logging.info(f"lr={lr}, head={self.heads}")
        cost = torch.nn.CrossEntropyLoss().to(device=self.device)
        tids = []
        ratio_full = self.config['ratio_full']
        ratio_ns2 = self.config['ratio_ns2'] if ns_num else 0
        ratio_ns1 = self.config['ratio_ns1'] if ns_num else 0
        ratio_ps1 = self.config['ratio_ps1'] if ps_num else 0
        ratio_ps2 = self.config['ratio_ps2'] if ps_num else 0

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = [], [], [], [], []
            for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2, batch_tid in tqdm(data_loader):  # 每个批次循环
                tids += batch_tid
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

                # full_loss: A -- B random sample
                label = torch.tensor([i for i in range(len(tid1))]).to(device=self.device)
                full_loss = 0
                if ratio_full:
                    sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                    full_loss = cost(sim1, label)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    sim2 = []
                    for ind, i in enumerate(ns2):
                        ns2_vec = x[i]
                        sim2.append(torch.matmul(tid1_vec[ind], ns2_vec.T))
                    sim2 = torch.stack(sim2)
                    ns2_loss = cost(sim2, label)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    sim3 = []
                    for ind, i in enumerate(ns1):
                        ns1_vec = x[i]
                        sim3.append(torch.matmul(tid2_vec[ind], ns1_vec.T))
                    sim3 = torch.stack(sim3)
                    ns1_loss = cost(sim3, label)
                    epoch_ns1.append(ns1_loss.data.item())

                # ps1_loss: enhance positive A --- enhance negative B
                ps1_loss = 0
                if ratio_ps1:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps1]
                        ps1_col_vec = x[columns]
                        sim4 = []
                        for ind, j in enumerate(ns2):
                            ns2_vec = x[j]
                            sim4.append(torch.matmul(ps1_col_vec[ind], ns2_vec.T))  # ns2_vec   tid2_vec
                        sim4 = torch.stack(sim4)
                        ps1_loss += cost(sim4, label)
                    epoch_ps1.append(ps1_loss.data.item())

                # loo5: enhance positive B --- enhance negative A
                ps2_loss = 0
                if ratio_ps2:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps2]
                        ps2_col_vec = x[columns]
                        sim5 = []
                        for ind, j in enumerate(ns2):
                            ns1_vec = x[j]
                            sim5.append(torch.matmul(ps2_col_vec[ind], ns1_vec.T))  # ns1_vec   tid1_vec
                        sim5 = torch.stack(sim5)
                        ps2_loss += cost(sim5, label)
                    epoch_ps2.append(ps2_loss.data.item())

                # loss = 50*full_loss + 10*ns2_loss + 10*ns1_loss + ps1_loss + ps2_loss
                loss = ratio_full * full_loss + ratio_ns2 * ns2_loss + ratio_ns1 * ns1_loss + ratio_ps1 * ps1_loss + ratio_ps2 * ps2_loss
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f"{full_loss} + {ns2_loss} + {ns1_loss} + {ps1_loss} + {ps2_loss} = {loss.data.item()}")
            torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')

            epoch_loss = epoch_loss / len(data_loader)
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = sorted(epoch_full), sorted(epoch_ns2), sorted(epoch_ns1), sorted(epoch_ps1), sorted(epoch_ps2)
            logging.info(f"epoch loss:{epoch_loss}")
            if ratio_full:
                mean_full = sum(epoch_full)/len(epoch_full)
                logging.info(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")
                print(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")

            if ratio_ns1:
                mean_ns1 = sum(epoch_ns1)/len(epoch_ns1)
                logging.info(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
                print(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
            if ratio_ns2:
                mean_ns2 = sum(epoch_ns2)/len(epoch_ns2)
                logging.info(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
                print(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
            if ratio_ps1:
                mean_ps1 = sum(epoch_ps1)/len(epoch_ps1)
                logging.info(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
                print(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
            if ratio_ps2:
                mean_ps2 = sum(epoch_ps2)/len(epoch_ps2)
                logging.info(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")
                print(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")

            if test_tid is not None:
                self.infer(test_tid, net_name=self.net_name)

    def ab_infoNCE_loss(self, train_tid, enhance_tid, test_tid=None):
        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        self.heads = 4
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        lr = self.config['lr']  # 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()

        logging.info(f"lr={lr}, head={self.heads}")
        tids = []
        ratio_full = self.config['ratio_full']
        ratio_ns2 = self.config['ratio_ns2'] if ns_num else 0
        ratio_ns1 = self.config['ratio_ns1'] if ns_num else 0
        ratio_ps1 = self.config['ratio_ps1'] if ps_num else 0
        ratio_ps2 = self.config['ratio_ps2'] if ps_num else 0

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = [], [], [], [], []
            for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2, batch_tid in tqdm(
                    data_loader):  # 每个批次循环
                tids += batch_tid
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

                # full_loss: A -- B random sample
                full_loss = 0
                if ratio_full:
                    full_loss = infoNCE_loss(tid1_vec, tid2_vec)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    for ind, i in enumerate(ns2):
                        ns2_vec = x[i]
                        ns2_loss += infoNCE_loss(tid1_vec[ind].repeat(len(tid1), 1), ns2_vec)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    for ind, i in enumerate(ns1):
                        ns1_vec = x[i]
                        ns1_loss += infoNCE_loss(tid2_vec[ind].repeat(len(tid1), 1), ns1_vec)
                    epoch_ns1.append(ns1_loss.data.item())

                # ps1_loss: enhance positive A --- enhance negative B
                ps1_loss = 0
                if ratio_ps1:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps1]
                        ps1_col_vec = x[columns]
                        for ind, j in enumerate(ns2):
                            ns2_vec = x[j]
                            ps1_loss += infoNCE_loss(ps1_col_vec[ind].repeat(len(tid1), 1), ns2_vec)
                    epoch_ps1.append(ps1_loss.data.item())

                # loo5: enhance positive B --- enhance negative A
                ps2_loss = 0
                if ratio_ps2:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps2]
                        ps2_col_vec = x[columns]
                        for ind, j in enumerate(ns2):
                            ns1_vec = x[j]
                            ps2_loss += infoNCE_loss(ps2_col_vec[ind].repeat(len(tid1), 1), ns1_vec)
                    epoch_ps2.append(ps2_loss.data.item())

                # loss = 50*full_loss + 10*ns2_loss + 10*ns1_loss + ps1_loss + ps2_loss
                loss = ratio_full * full_loss + ratio_ns2 * ns2_loss + ratio_ns1 * ns1_loss + ratio_ps1 * ps1_loss + ratio_ps2 * ps2_loss
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f"{full_loss} + {ns2_loss} + {ns1_loss} + {ps1_loss} + {ps2_loss} = {loss.data.item()}")
            torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')

            epoch_loss = epoch_loss / len(data_loader)
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = sorted(epoch_full), sorted(epoch_ns2), sorted(
                epoch_ns1), sorted(epoch_ps1), sorted(epoch_ps2)
            logging.info(f"epoch loss:{epoch_loss}")
            if ratio_full:
                mean_full = sum(epoch_full) / len(epoch_full)
                logging.info(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")
                print(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")

            if ratio_ns1:
                mean_ns1 = sum(epoch_ns1) / len(epoch_ns1)
                logging.info(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
                print(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
            if ratio_ns2:
                mean_ns2 = sum(epoch_ns2) / len(epoch_ns2)
                logging.info(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
                print(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
            if ratio_ps1:
                mean_ps1 = sum(epoch_ps1) / len(epoch_ps1)
                logging.info(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
                print(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
            if ratio_ps2:
                mean_ps2 = sum(epoch_ps2) / len(epoch_ps2)
                logging.info(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")
                print(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")

            if test_tid is not None:
                self.infer(test_tid, net_name=self.net_name)

    def ab_triplet_margin_loss(self, train_tid, enhance_tid, test_tid=None):   # TripletMarginLoss
        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        self.heads = 4
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        lr = self.config['lr']  # 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()

        logging.info(f"lr={lr}, head={self.heads}")
        cost = torch.nn.TripletMarginLoss().to(device=self.device)
        tids = []
        ratio_full = self.config['ratio_full']
        ratio_ns2 = self.config['ratio_ns2'] if ns_num else 0
        ratio_ns1 = self.config['ratio_ns1'] if ns_num else 0
        ratio_ps1 = self.config['ratio_ps1'] if ps_num else 0
        ratio_ps2 = self.config['ratio_ps2'] if ps_num else 0

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = [], [], [], [], []
            for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2, batch_tid in tqdm(data_loader):  # 每个批次循环
                tids += batch_tid
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

                # full_loss: A -- B random sample
                full_loss = 0
                if ratio_full:
                    negative = torch.cat((tid2_vec[-1:], tid2_vec[:-1])).to(device=self.device)
                    full_loss = cost(tid1_vec, tid2_vec, negative)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    for i in range(self.config['ns_num']):
                        column = [row[i] for row in ns2]
                        column[i] = column[i - 1]
                        negative = x[column]
                        ns2_loss += cost(tid1_vec, tid2_vec, negative)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    for i in range(self.config['ns_num']):
                        column = [row[i] for row in ns1]
                        column[i] = column[i - 1]
                        negative = x[column]
                        ns1_loss += cost(tid2_vec, tid1_vec, negative)
                    epoch_ns1.append(ns1_loss.data.item())

                # ps1_loss: enhance positive A --- enhance negative B
                ps1_loss = 0
                if ratio_ps1:
                    for i in range(self.config['ps_num']):
                        column = [row[i] for row in ps1]
                        archor = x[column]
                        negative = torch.cat((tid2_vec[-1:], tid2_vec[:-1])).to(device=self.device)
                        ps1_loss += cost(archor, tid2_vec, negative)
                    epoch_ps1.append(ps1_loss.data.item())

                # loo5: enhance positive B --- enhance negative A
                ps2_loss = 0
                if ratio_ps2:
                    for i in range(self.config['ps_num']):
                        column = [row[i] for row in ps2]
                        archor = x[column]
                        negative = torch.cat((tid1_vec[-1:], tid1_vec[:-1])).to(device=self.device)
                        ps2_loss += cost(archor, tid1_vec, negative)
                    epoch_ps2.append(ps2_loss.data.item())

                # loss = 50*full_loss + 10*ns2_loss + 10*ns1_loss + ps1_loss + ps2_loss
                loss = ratio_full * full_loss + ratio_ns2 * ns2_loss + ratio_ns1 * ns1_loss + ratio_ps1 * ps1_loss + ratio_ps2 * ps2_loss
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f"{full_loss} + {ns2_loss} + {ns1_loss} + {ps1_loss} + {ps2_loss} = {loss.data.item()}")
            torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')

            epoch_loss = epoch_loss / len(data_loader)
            epoch_full, epoch_ns2, epoch_ns1, epoch_ps1, epoch_ps2 = sorted(epoch_full), sorted(epoch_ns2), sorted(epoch_ns1), sorted(epoch_ps1), sorted(epoch_ps2)
            logging.info(f"epoch loss:{epoch_loss}")
            if ratio_full:
                mean_full = sum(epoch_full)/len(epoch_full)
                logging.info(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")
                print(f"full_loss min:{epoch_full[:2]}, max:{epoch_full[-1]}, mean:{mean_full}")

            if ratio_ns1:
                mean_ns1 = sum(epoch_ns1)/len(epoch_ns1)
                logging.info(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
                print(f"ns1_loss min:{epoch_ns1[:2]}, max:{epoch_ns1[-1]}, mean:{mean_ns1}")
            if ratio_ns2:
                mean_ns2 = sum(epoch_ns2)/len(epoch_ns2)
                logging.info(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
                print(f"ns2_loss min:{epoch_ns2[:2]}, max:{epoch_ns2[-1]}, mean:{mean_ns2}")
            if ratio_ps1:
                mean_ps1 = sum(epoch_ps1)/len(epoch_ps1)
                logging.info(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
                print(f"ps1_loss min:{epoch_ps1[:2]}, max:{epoch_ps1[-1]}, mean:{mean_ps1}")
            if ratio_ps2:
                mean_ps2 = sum(epoch_ps2)/len(epoch_ps2)
                logging.info(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")
                print(f"ps2_loss min:{epoch_ps2[:2]}, max:{epoch_ps2[-1]}, mean:{mean_ps2}")

            if test_tid is not None:
                self.infer(test_tid, net_name=self.net_name)
