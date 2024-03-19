import logging
import os
from random import sample

import numpy as np
from libTrajectory.model.STEL import GCN
from libTrajectory.preprocessing.STEL.graphDataset import GraphDataset
from libTrajectory.evaluator.faiss_cosine import evaluator
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

log_path = "./libTrajectory/logs/STEL/"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Executor(object):
    def __init__(self, path, stid_counts, mid_dim=128, out_dim=128):
        self.path = path
        self.stid_counts = stid_counts
        logging.info(f"Executor...")
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_dim = 8 + 17 + 15 + 16
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        logging.info(f"device={self.device}, in_dim={self.in_dim}, mid_dim={self.mid_dim}, out_dim={self.out_dim}")

    def batch_struct(self, data):
        # data[key] = (node, edge_ind, edge_attr) or None
        struct = {'tid1': {}, 'tid2': {}, 'ps1': {}, 'ps2': {}, 'ns1': {}, 'ns2': {}}
        # g1
        node1 = []
        edge_ind1 = [[], []]
        edge_attr1 = []
        # g2
        node2 = []
        edge_ind2 = [[], []]
        edge_attr2 = []
        # ns is None: random generate ns
        random_ns1 = []
        random_ns2 = []

        for ind, dic in enumerate(data):
            add_len1 = len(node1)
            node1 = node1 + dic['g1'][0]
            e_ind0, e_ind1 = dic['g1'][1]
            edge_ind1[0] += [i + add_len1 for i in e_ind0]
            edge_ind1[1] += [i + add_len1 for i in e_ind1]
            edge_attr1.extend(dic['g1'][2])
            struct['tid1'][ind] = add_len1

            add_len2 = len(node2)
            node2 = node2 + dic['g2'][0]
            e_ind0, e_ind1 = dic['g2'][1]
            edge_ind2[0] += [i + add_len2 for i in e_ind0]
            edge_ind2[1] += [i + add_len2 for i in e_ind1]
            edge_attr2.extend(dic['g2'][2])
            struct['tid2'][ind] = add_len2

            if dic['ps1'] is not None:
                add_len1 = len(node1)
                node1 = node1 + dic['ps1'][0]
                e_ind0, e_ind1 = dic['ps1'][1]
                edge_ind1[0] += [i + add_len1 for i in e_ind0]
                edge_ind1[1] += [i + add_len1 for i in e_ind1]
                edge_attr1.extend(dic['ps1'][2])
                struct['ps1'][ind] = add_len1

            if dic['ps2'] is not None:
                add_len2 = len(node2)
                node2 = node2 + dic['ps2'][0]
                e_ind0, e_ind1 = dic['ps2'][1]
                edge_ind2[0] += [i + add_len2 for i in e_ind0]
                edge_ind2[1] += [i + add_len2 for i in e_ind1]
                edge_attr2.extend(dic['ps2'][2])
                struct['ps2'][ind] = add_len2

            if dic['ns1'] is not None:
                add_len1 = len(node1)
                node1 = node1 + dic['ns1'][0]
                e_ind0, e_ind1 = dic['ns1'][1]
                edge_ind1[0] += [i + add_len1 for i in e_ind0]
                edge_ind1[1] += [i + add_len1 for i in e_ind1]
                edge_attr1.extend(dic['ns1'][2])
                struct['ns1'][ind] = add_len1
            else:
                random_ns1.append(ind)

            if dic['ns2'] is not None:
                add_len2 = len(node2)
                node2 = node2 + dic['ns2'][0]
                e_ind0, e_ind1 = dic['ns2'][1]
                edge_ind2[0] += [i + add_len2 for i in e_ind0]
                edge_ind2[1] += [i + add_len2 for i in e_ind1]
                edge_attr2.extend(dic['ns2'][2])
                struct['ns2'][ind] = add_len2
            else:
                random_ns2.append(ind)

        for ind in random_ns1:
            while True:
                random_ind = sample(list(struct['tid1'].keys()), 1)[0]
                if random_ind != ind:
                    break
            struct['ns1'][ind] = random_ind

        for ind in random_ns2:
            while True:
                random_ind = sample(list(struct['tid2'].keys()), 1)[0]
                if random_ind != ind:
                    break
            struct['ns2'][ind] = random_ind
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

        return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2, struct

    def train(self, train_tid, enhance_ns, test_tid=None, epoch_num=1, batch_size=32, num_workers=48):
        logging.info(f"train, epoch_num={epoch_num}, batch_size={batch_size}, num_workers={num_workers}")
        graph_data = GraphDataset(self.path, train_tid, self.stid_counts, train=True, enhance_ns=enhance_ns)
        data_loader = DataLoader(dataset=graph_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 collate_fn=lambda x: x, persistent_workers=True)
        # net1
        net1 = GCN(in_dim=self.in_dim, out_dim=self.mid_dim, device=self.device).to(self.device)
        lr1 = 0.001
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr1)
        net1.train()
        # net2
        net2 = GCN(in_dim=self.in_dim, out_dim=self.mid_dim, device=self.device).to(self.device)
        lr2 = 0.001
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr2)
        net2.train()
        # net3
        net3 = GCN(in_dim=self.mid_dim, out_dim=self.out_dim, device=self.device).to(self.device)
        lr3 = 0.001
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=lr3)
        net3.train()
        logging.info(f"lr1={lr1}, lr2={lr2}, lr3={lr3}")

        cost = torch.nn.CrossEntropyLoss().to(self.device)
        for epoch in range(epoch_num):  # 每个epoch循环
            logging.info(f'Epoch {epoch}/{epoch_num}')
            batch_loss = 0
            tid_len = len(train_tid)
            num = 1
            for data in data_loader:  # 每个批次循环
                num = num*batch_size
                node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2, struct = self.batch_struct(data)

                # 前向传播
                net1_x = net1(node1, edge_ind1, edge_attr1)
                net2_x = net2(node2, edge_ind2, edge_attr2)
                x1 = net3(net1_x, edge_ind1, edge_attr1)
                x2 = net3(net2_x, edge_ind2, edge_attr2)

                # 计算损失
                # 增强正样本 loss1  "enh1": [[A A'], ...]
                label = []
                vector1 = []
                for label_num, ind in enumerate(struct['ps1']):
                    vector1.append(x1[struct['tid1'][ind]])
                    vector1.append(x1[struct['ps1'][ind]])
                    label.append(label_num)
                    label.append(label_num)
                label_enh1 = len(struct['ps1'])

                for label_num, ind in enumerate(struct['ps2']):
                    vector1.append(x2[struct['tid2'][ind]])
                    vector1.append(x2[struct['ps2'][ind]])
                    label.append(label_num + label_enh1)
                    label.append(label_num + label_enh1)
                label = torch.tensor(label)
                vector = torch.stack(vector1, dim=0)
                if 'cuda' in str(self.device):
                    label = label.to(device=self.device)
                    vector = vector.to(device=self.device)
                loss1 = cost(vector, label)

                # 正样本与负样本
                label = []
                vector1 = []
                vector2 = []
                for ind in struct['tid1']:  # [A B]
                    vector1.append(x1[struct['tid1'][ind]])
                    vector2.append(x2[struct['tid2'][ind]])
                    label.append(1)
                    vector1.append(x1[struct['tid1'][ind]])
                    vector2.append(x2[struct['ns2'][ind]])
                    label.append(0)
                    vector2.append(x2[struct['tid2'][ind]])
                    vector1.append(x1[struct['ns1'][ind]])
                    label.append(0)

                label = torch.tensor(label, dtype=torch.float32)
                vector1 = torch.stack(vector1, dim=0)
                vector2 = torch.stack(vector2, dim=0)
                if 'cuda' in str(self.device):
                    label = label.to(device=self.device)
                    vector1 = vector1.to(device=self.device)
                    vector2 = vector2.to(device=self.device)
                similarities = F.cosine_similarity(vector1, vector2)
                loss2 = cost(similarities, label)

                loss = loss1 + loss2
                # 反向传播
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                logging.info(f"bar={num}/{tid_len}, loss1:{loss1} + loss2:{loss2} = {loss.data.item()}")
                batch_loss += loss.data.item()

            epoch_loss = batch_loss / len(data_loader)
            logging.info(f"epoch loss: {epoch_loss}")
            torch.save(net1.state_dict(), f'{log_path}net1_parameter-epoch:{epoch}.pth')
            torch.save(net2.state_dict(), f'{log_path}net2_parameter-epoch:{epoch}.pth')
            torch.save(net3.state_dict(), f'{log_path}net3_parameter-epoch:{epoch}.pth')
            if test_tid is not None:
                self.infer(test_tid, para1=f'net1_parameter-epoch:{epoch}.pth',
                           para2=f'net2_parameter-epoch:{epoch}.pth',
                           para3=f'net3_parameter-epoch:{epoch}.pth')
        torch.save(net1.state_dict(), f'{log_path}net1_parameter.pth')
        torch.save(net2.state_dict(), f'{log_path}net2_parameter.pth')
        torch.save(net3.state_dict(), f'{log_path}net3_parameter.pth')

    def infer(self, test_data, para1='net1_parameter-epoch:0.pth', para2='net2_parameter-epoch:0.pth', para3='net3_parameter-epoch:0.pth'):
        logging.info("test")
        state_dict1 = torch.load(f'{log_path}{para1}')
        net1 = GCN(in_dim=self.in_dim, out_dim=self.mid_dim, device=self.device).to(self.device)
        net1.load_state_dict(state_dict1)
        net1.eval()

        state_dict2 = torch.load(f'{log_path}{para2}')
        net2 = GCN(in_dim=self.in_dim, out_dim=self.mid_dim, device=self.device).to(self.device)
        net2.load_state_dict(state_dict2)
        net2.eval()

        state_dict3 = torch.load(f'{log_path}{para3}')
        net3 = GCN(in_dim=self.mid_dim, out_dim=self.out_dim, device=self.device).to(self.device)
        net3.load_state_dict(state_dict3)
        net3.eval()
        logging.info(f"net1={para1}, net2={para2}, net3={para3}")

        for k, v in test_data.items():
            logging.info(f"{k}...")
            graph_data = GraphDataset(self.path, v, self.stid_counts, train=False)  # , self.st_vec
            data_loader = DataLoader(graph_data, batch_size=2, num_workers=8, collate_fn=lambda x: x, persistent_workers=True)
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



