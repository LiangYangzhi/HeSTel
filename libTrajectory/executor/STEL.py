import logging
import pickle

import numpy as np
from tqdm import tqdm

from libTrajectory.model.hst import HST
from libTrajectory.model.ab_hst_net import GCN, GTransformer, GAT
from libTrajectory.model.GATModel import GAT as baselineGAT
from libTrajectory.model.GCNModel import GCN1 as baselineGCN, TwinTowerGCN
from libTrajectory.model.GraphModel import Graph1 as baselineGraph
from libTrajectory.model.LSTMModel import LSTM1 as baselineLSTM
from libTrajectory.model.transformerModel import Transformer as baselineTransformer

from libTrajectory.model.loss import decision_loss, infoNCE_loss
from libTrajectory.preprocessing.STEL.baseline import SignaturePre
from libTrajectory.preprocessing.STEL.batch_collate import train_coll, infer_coll, baseline_single_coll, \
    baseline_twin_coll, baseline_seq_coll
from libTrajectory.preprocessing.STEL.graphDataset import GraphLoader
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader


class Executor(object):
    def __init__(self, log_path, config):
        logging.info(f"Executor...")
        self.path = config['path']
        self.log_path = log_path
        self.config = config['executor']

        cuda = self.config['cuda']
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.in_dim = self.config['in_dim']
        self.out_dim = self.config['out_dim']
        self.net_name = self.config['net_name']
        self.heads = self.config['head']
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
        elif self.config.get("loss") == "CosineEmbedding":
            self.ab_cosine_embedding_loss(train_tid, enhance_tid, test_tid=test_tid)
            return None

        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num, self.config.get("graph"))
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        if self.config.get("net") == "GCN":
            net = GCN(self.in_dim, self.out_dim*self.heads).to(self.device)
        elif self.config.get("net") == "GAT":
            net = GAT(self.in_dim, self.out_dim, self.heads).to(self.device)
        elif self.config.get("net") == "GTransformer":
            net = GTransformer(self.in_dim, self.out_dim, self.heads).to(self.device)
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
                    full_loss = decision_loss(sim1, penalty=penalty)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    sim2 = []
                    for ind, i in enumerate(ns2):
                        ns2_vec = x[i]
                        sim2.append(torch.matmul(tid1_vec[ind], ns2_vec.T))
                    sim2 = torch.stack(sim2)
                    ns2_loss = decision_loss(sim2, penalty=penalty)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    sim3 = []
                    for ind, i in enumerate(ns1):
                        ns1_vec = x[i]
                        sim3.append(torch.matmul(tid2_vec[ind], ns1_vec.T))
                    sim3 = torch.stack(sim3)
                    ns1_loss = decision_loss(sim3, penalty=penalty)
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
                        ps1_loss += decision_loss(sim4, penalty=penalty)
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
                        ps2_loss += decision_loss(sim5, penalty=penalty)
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
        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
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
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
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

    def ab_cosine_embedding_loss(self, train_tid, enhance_tid, test_tid=None):
        logging.info(f"train...")
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        ps_num = self.config['ps_num']
        ns_num = self.config['ns_num']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()

        logging.info(f"lr={lr}, head={self.heads}")
        cost = torch.nn.CosineEmbeddingLoss().to(device=self.device)
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
                    size = tid1_vec.shape[0]
                    for i in range(size):
                        label = -torch.ones(size).to(device=self.device)
                        label[i] = 1
                        embedding1 = tid1_vec[i].repeat(size, 1)
                        full_loss += cost(embedding1, tid2_vec, label)
                    epoch_full.append(full_loss.data.item())

                # ns2_loss: A --- enhance negative B
                ns2_loss = 0
                if ratio_ns2:
                    size = tid1_vec.shape[0]
                    for ind, i in enumerate(ns2):
                        ns2_vec = x[i]
                        label = -torch.ones(size).to(device=self.device)
                        label[ind] = 1
                        embedding1 = tid1_vec[ind].repeat(size, 1)
                        ns2_loss += cost(embedding1, ns2_vec, label)
                    epoch_ns2.append(ns2_loss.data.item())

                # ns1_loss: B --- enhance negative A
                ns1_loss = 0
                if ratio_ns1:
                    size = tid1_vec.shape[0]
                    for ind, i in enumerate(ns1):
                        ns1_vec = x[i].to(device=self.device)
                        label = -torch.ones(size).to(device=self.device)
                        label[ind] = 1
                        embedding2 = tid2_vec[ind].repeat(size, 1)
                        ns1_loss += cost(embedding2, ns1_vec, label)
                    epoch_ns1.append(ns1_loss.data.item())

                # ps1_loss: enhance positive A --- enhance negative B
                ps1_loss = 0
                if ratio_ps1:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps1]
                        ps1_col_vec = x[columns]
                        size = tid1_vec.shape[0]
                        for ind, j in enumerate(ns2):
                            ns2_vec = x[j].to(device=self.device)
                            label = -torch.ones(size).to(device=self.device)
                            label[ind] = 1
                            embedding1 = ps1_col_vec[ind].repeat(size, 1)
                            ps1_loss += cost(embedding1, ns2_vec, label)
                    epoch_ps1.append(ps1_loss.data.item())

                # loo5: enhance positive B --- enhance negative A
                ps2_loss = 0
                if ratio_ps2:
                    for i in range(ps_num):
                        columns = [row[i] for row in ps2]
                        ps2_col_vec = x[columns]
                        size = tid1_vec.shape[0]
                        for ind, j in enumerate(ns2):
                            ns1_vec = x[j].to(device=self.device)
                            label = -torch.ones(size).to(device=self.device)
                            label[ind] = 1
                            embedding2 = ps2_col_vec[ind].repeat(size, 1)
                            ps2_loss += cost(embedding2, ns1_vec, label)
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
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
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
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, enhance_tid, ps_num, ns_num)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=train_coll, persistent_workers=True, shuffle=True)
        # net
        net = HST(self.in_dim, self.out_dim, self.heads).to(self.device)
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


class BLExecutor(object):
    def __init__(self, log_path, config):
        logging.info(f"Executor...")
        self.path = config['path']
        self.log_path = log_path
        self.config = config['executor']

    # signature: sequence, spatial, temporal, spatiotemporal
    def signature(self, config):
        """
        << Trajectory-Based Spatiotemporal Entity Linking >> 实验复现

        signature
        sequential signature：时空点作为词，grams设置为2，进行TF-IDF提取向量并进行L2 normalization
        temporal signature：一天中的1h作为时间间隔，统计在每个时间间隔内出现的频率并进行L1 normalization
        spatial signature：空间点作为词，进行TF-IDF提取向量进行L2 normalizations
        spatiotemporal signature：时间间隔+空间点作为词，进行TF-IDF提取向量进行L2 normalization。

        similarity
        sequential similarity：dot product
        temporal similarity：(1- EMD) distance
        spatial similarity: dot product
        spatiotemporal similarity: dot product

        base knn query
        """
        preprocessor = SignaturePre(config)

        # 序列signature
        test_data = preprocessor.sequential()
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            evaluator(v1, v2)

        # 时间signature
        temporal_config = self.config["temporal"]  # ['year_month', 'month_day', 'week_day', 'day_hour']
        test_data = preprocessor.temporal(method=temporal_config['method'])
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            evaluator(v1, v2)

        # 空间signature
        test_data = preprocessor.spatial()
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            evaluator(v1, v2)

        # 时空signature
        spatiotemporal_config = self.config["spatiotemporal"]  # ['year_month', 'month_day', 'week_day', 'day_hour']
        test_data = preprocessor.spatiotemporal(method=spatiotemporal_config['method'])
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            evaluator(v1, v2)

    def gcn_infer(self, test_data):
        logging.info("test")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']

        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
        net = baselineGCN(in_dim, out_dim).to(device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=batch_size, num_workers=num_workers,
                                     collate_fn=baseline_single_coll, persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, edge, edge_attr, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)
                x = net(node, edge, edge_attr)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

    def gcn_cross(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGCN(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CrossEntropyLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)
                label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss = cost(sim1, label)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

        if test_tid is not None:
            self.gcn_infer(test_tid)

    def gcn_cosine(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGCN(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CosineEmbeddingLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)

                size = tid1_vec.shape[0]
                loss = 0
                for i in range(size):
                    label = -torch.ones(size).to(device=device)
                    label[i] = 1
                    embedding1 = tid1_vec[i].repeat(size, 1)
                    loss += cost(embedding1, tid2_vec, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.gcn_infer(test_tid)

    def graph_infer(self, test_data):
        logging.info("test")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']

        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
        net = baselineGraph(in_dim, out_dim).to(device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=batch_size, num_workers=num_workers,
                                     collate_fn=baseline_single_coll, persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, edge, edge_attr, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)
                x = net(node, edge, edge_attr)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

    def graph_cross(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGraph(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CrossEntropyLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)
                label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss = cost(sim1, label)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.graph_infer(test_tid)

    def graph_cosine(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, None, [], 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGraph(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CosineEmbeddingLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)

                size = tid1_vec.shape[0]
                loss = 0
                for i in range(size):
                    label = -torch.ones(size).to(device=device)
                    label[i] = 1
                    embedding1 = tid1_vec[i].repeat(size, 1)
                    loss += cost(embedding1, tid2_vec, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.graph_infer(test_tid)

    def gTransformer_infer(self, test_data):
        logging.info("test")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']

        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
        net = baselineGAT(in_dim, out_dim, heads).to(device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=16, num_workers=2, collate_fn=baseline_single_coll,
                                     persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, edge, edge_attr, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)
                x = net(node, edge, edge_attr)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

    def gTransformer_cross(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGAT(in_dim, out_dim, heads).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CrossEntropyLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)
                label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss = cost(sim1, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.gTransformer_infer(test_tid)

    def gTransformer_cosine(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_single_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineGAT(in_dim, out_dim, heads).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CosineEmbeddingLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, edge, edge_attr, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                    edge = edge.to(device=device)
                    edge_attr = edge_attr.to(device=device)

                # 前向传播
                x = net(node, edge, edge_attr)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)

                size = tid1_vec.shape[0]
                loss = 0
                for i in range(size):
                    label = -torch.ones(size).to(device=device)
                    label[i] = 1
                    embedding1 = tid1_vec[i].repeat(size, 1)
                    loss += cost(embedding1, tid2_vec, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.gTransformer_infer(test_tid)

    def lstm_infer(self, test_data):
        logging.info("test")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
        net = baselineLSTM(in_dim, out_dim).to(device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=16, num_workers=2, collate_fn=baseline_seq_coll,
                                     persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                x = net(node)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

    def lstm_cross(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_seq_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineLSTM(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CrossEntropyLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)

                # 前向传播
                x = net(node)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)
                label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss = cost(sim1, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.lstm_infer(test_tid)

    def lstm_cosine(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_seq_coll, persistent_workers=True, shuffle=True)
        # net
        net = baselineLSTM(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CosineEmbeddingLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)

                # 前向传播
                x = net(node)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)

                size = tid1_vec.shape[0]
                loss = 0
                for i in range(size):
                    label = -torch.ones(size).to(device=device)
                    label[i] = 1
                    embedding1 = tid1_vec[i].repeat(size, 1)
                    loss += cost(embedding1, tid2_vec, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.lstm_infer(test_tid)

    def transformer_infer(self, test_data):
        logging.info("test")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
        if "taxi" in self.path:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads, num_layers=14).to(device)
        else:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads).to(device)
        net.load_state_dict(state_dict1)
        net.eval()
        logging.info(f"net={net_name}")

        for k, v in test_data.items():  # file_name: tid
            logging.info(f"{k}...")
            print(f"{k}...")
            graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=16, num_workers=2, collate_fn=baseline_seq_coll,
                                     persistent_workers=True)
            embedding_1 = []
            embedding_2 = []
            for node, tid1, tid2 in data_loader:  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)
                x = net(node)
                vec1 = x[tid1]
                vec2 = x[tid2]
                if 'cuda' in str(device):
                    vec1 = vec1.to(device='cpu')
                    vec2 = vec2.to(device='cpu')

                for i in vec1:
                    embedding_1.append(i.detach().numpy())
                for i in vec2:
                    embedding_2.append(i.detach().numpy())
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)

    def transformer_cross(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_seq_coll, persistent_workers=True, shuffle=True)
        # net
        if "taxi" in self.path:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads, num_layers=14).to(device)
        else:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CrossEntropyLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)

                # 前向传播
                x = net(node)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)
                label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
                sim1 = torch.matmul(tid1_vec, tid2_vec.T)
                loss = cost(sim1, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.transformer_infer(test_tid)

    def transformer_cosine(self, train_tid, test_tid=None):
        logging.info(f"train...")
        cuda = self.config['cuda']
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        in_dim = self.config['in_dim']
        heads = self.config['head']
        out_dim = self.config['out_dim']
        net_name = self.config['net_name']
        epoch_num = self.config['epoch_num']
        num_workers = self.config['num_workers']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=baseline_seq_coll, persistent_workers=True, shuffle=True)
        # net
        if "taxi" in self.path:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads, num_layers=14).to(device)
        else:
            net = baselineTransformer(input_dim=in_dim, d_model=out_dim, nhead=heads).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        cost = torch.nn.CosineEmbeddingLoss().to(device=device)

        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            epoch_loss = 0
            for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
                if 'cuda' in str(device):
                    node = node.to(device=device)

                # 前向传播
                x = net(node)
                tid1_vec = x[tid1].to(device=device)
                tid2_vec = x[tid2].to(device=device)

                size = tid1_vec.shape[0]
                loss = 0
                for i in range(size):
                    label = -torch.ones(size).to(device=device)
                    label[i] = 1
                    embedding1 = tid1_vec[i].repeat(size, 1)
                    loss += cost(embedding1, tid2_vec, label)
                epoch_loss += loss.data.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(loss.data.item())

            torch.save(net.state_dict(), f'{self.log_path}{net_name}.pth')
            epoch_loss = epoch_loss / len(data_loader)
            logging.info(f"epoch loss:{epoch_loss}")

            if test_tid is not None:
                self.transformer_infer(test_tid)

    # def twin_gcn_infer(self, test_data, net_name=None):
    #     logging.info("test")
    #     if net_name is None:
    #         net_name = self.net_name
    #     state_dict1 = torch.load(f'{self.log_path}{net_name}.pth', map_location='cpu')
    #     net = TwinTowerGCN(self.in_dim, self.out_dim*self.heads).to(self.device)
    #     net.load_state_dict(state_dict1)
    #     net.eval()
    #     logging.info(f"net={self.net_name}")
    #
    #     for k, v in test_data.items():  # file_name: tid
    #         logging.info(f"{k}...")
    #         print(f"{k}...")
    #         graph_data = GraphLoader(self.path, v, train=False)  # , self.st_vec
    #         data_loader = DataLoader(graph_data, batch_size=16, num_workers=12, collate_fn=baseline_twin_coll,
    #                                  persistent_workers=True)
    #         embedding_1 = []
    #         embedding_2 = []
    #         for node1, edge1, edge_attr1, tid1, node2, edge2, edge_attr2, tid2 in tqdm(data_loader):  # 每个批次循环
    #             if 'cuda' in str(self.device):
    #                 node1 = node1.to(device=self.device)
    #                 edge1 = edge1.to(device=self.device)
    #                 edge_attr1 = edge_attr1.to(device=self.device)
    #                 node2 = node2.to(device=self.device)
    #                 edge2 = edge2.to(device=self.device)
    #                 edge_attr2 = edge_attr2.to(device=self.device)
    #             x1, x2 = net(node1, edge1, edge_attr1, node2, edge2, edge_attr2)
    #             vec1 = x1[tid1]
    #             vec2 = x2[tid2]
    #             if 'cuda' in str(self.device):
    #                 vec1 = vec1.to(device='cpu')
    #                 vec2 = vec2.to(device='cpu')
    #
    #             for i in vec1:
    #                 embedding_1.append(i.detach().numpy())
    #             for i in vec2:
    #                 embedding_2.append(i.detach().numpy())
    #         embedding_1 = np.array(embedding_1)
    #         embedding_2 = np.array(embedding_2)
    #         evaluator(embedding_1, embedding_2)
    #
    # def twin_gcn_cross_entropy(self, train_tid, test_tid=None):
    #     logging.info(f"train...")
    #     epoch_num = self.config['epoch_num']
    #     num_workers = self.config['num_workers']
    #     batch_size = self.config['batch_size']
    #     graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
    #     data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
    #                              collate_fn=baseline_twin_coll, persistent_workers=True, shuffle=True)
    #     # net
    #     net = TwinTowerGCN(self.in_dim, self.out_dim*self.heads).to(self.device)
    #     lr = self.config['lr']  # 0.001
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #     net.train()
    #
    #     logging.info(f"lr={lr}, head={self.heads}")
    #     cost = torch.nn.CrossEntropyLoss().to(device=self.device)
    #
    #     for epoch in range(epoch_num):  # 每个epoch循环
    #         logging.info(f'Epoch {epoch}/{epoch_num}')
    #         epoch_loss = 0
    #         for node1, edge1, edge_attr1, tid1, node2, edge2, edge_attr2, tid2 in tqdm(data_loader):  # 每个批次循环
    #             if 'cuda' in str(self.device):
    #                 node1 = node1.to(device=self.device)
    #                 edge1 = edge1.to(device=self.device)
    #                 edge_attr1 = edge_attr1.to(device=self.device)
    #                 node2 = node2.to(device=self.device)
    #                 edge2 = edge2.to(device=self.device)
    #                 edge_attr2 = edge_attr2.to(device=self.device)
    #             x1, x2 = net(node1, edge1, edge_attr1, node2, edge2, edge_attr2)
    #             tid1_vec = x1[tid1].to(device=self.device)
    #             tid2_vec = x2[tid2].to(device=self.device)
    #
    #             label = torch.tensor([i for i in range(len(tid1))]).to(device=self.device)
    #             sim1 = torch.matmul(tid1_vec, tid2_vec.T)
    #             loss = cost(sim1, label)
    #             epoch_loss += loss.data.item()
    #             # 反向传播
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             logging.info(loss.data.item())
    #
    #         torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')
    #         epoch_loss = epoch_loss / len(data_loader)
    #         logging.info(f"epoch loss:{epoch_loss}")
    #
    #         if test_tid is not None:
    #             self.twin_gcn_infer(test_tid, net_name=self.net_name)
    #
    # def twin_gcn_cosine_embedding(self, train_tid, test_tid=None):
    #     logging.info(f"train...")
    #     epoch_num = self.config['epoch_num']
    #     num_workers = self.config['num_workers']
    #     batch_size = self.config['batch_size']
    #     graph_data = GraphLoader(self.path, train_tid, True, None, 0, 0)
    #     data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, num_workers=num_workers,
    #                              collate_fn=baseline_twin_coll, persistent_workers=True, shuffle=True)
    #     # net
    #     net = TwinTowerGCN(self.in_dim, self.out_dim*self.heads).to(self.device)
    #     lr = self.config['lr']  # 0.001
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #     net.train()
    #
    #     logging.info(f"lr={lr}, head={self.heads}")
    #     cost = torch.nn.CosineEmbeddingLoss().to(device=self.device)
    #
    #     for epoch in range(epoch_num):  # 每个epoch循环
    #         logging.info(f'Epoch {epoch}/{epoch_num}')
    #         epoch_loss = 0
    #         for node1, edge1, edge_attr1, tid1, node2, edge2, edge_attr2, tid2 in tqdm(data_loader):  # 每个批次循环
    #             if 'cuda' in str(self.device):
    #                 node1 = node1.to(device=self.device)
    #                 edge1 = edge1.to(device=self.device)
    #                 edge_attr1 = edge_attr1.to(device=self.device)
    #                 node2 = node2.to(device=self.device)
    #                 edge2 = edge2.to(device=self.device)
    #                 edge_attr2 = edge_attr2.to(device=self.device)
    #             x1, x2 = net(node1, edge1, edge_attr1, node2, edge2, edge_attr2)
    #             tid1_vec = x1[tid1].to(device=self.device)
    #             tid2_vec = x2[tid2].to(device=self.device)
    #
    #             size = tid1_vec.shape[0]
    #             loss = 0
    #             for i in range(size):
    #                 label = -torch.ones(size).to(device=self.device)
    #                 label[i] = 1
    #                 embedding1 = tid1_vec[i].repeat(size, 1)
    #                 loss += cost(embedding1, tid2_vec, label)
    #             epoch_loss += loss.data.item()
    #             # 反向传播
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             logging.info(loss.data.item())
    #
    #         torch.save(net.state_dict(), f'{self.log_path}{self.net_name}.pth')
    #         epoch_loss = epoch_loss / len(data_loader)
    #         logging.info(f"epoch loss:{epoch_loss}")
    #
    #         if test_tid is not None:
    #             self.twin_gcn_infer(test_tid, net_name=self.net_name)
