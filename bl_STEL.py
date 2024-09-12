import logging
import pickle
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from libTrajectory.config.config_parser import parse_config
from libTrajectory.model.rnn import rnn as rnnModel
from libTrajectory.preprocessing.STEL.baseline import SignaturePre, NetPre, GraphLoader, rnn_coll
from libTrajectory.evaluator.knn_query import evaluator
# import torch

# lab 环境
# scikit-learn             1.3.0
# scipy                    1.11.2
# numpy                    1.25.2


def signature():
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
    test_data = preprocessor.temporal(method='day_hour')  # ['year_month', 'month_day', 'week_day', 'day_hour']
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        evaluator(v1, v2, method='day_hour')  # ['year_month', 'month_day', 'week_day', 'day_hour']

    # 空间signature
    test_data = preprocessor.spatial()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        evaluator(v1, v2)

    # 时空signature
    test_data = preprocessor.spatiotemporal(method='day_hour')  # ['year_month', 'month_day', 'week_day', 'day_hour']
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        evaluator(v1, v2)


def bl_rnn():
    train_tid, test_data, enhance_tid = NetPre(config).get(method='load')
    graph_data = GraphLoader(config['path'], train_tid)
    data_loader = DataLoader(dataset=graph_data, batch_size=config['executor']['batch_size'], num_workers=8,
                             collate_fn=rnn_coll, persistent_workers=True, shuffle=True)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    paramter = {"input_size": config['executor']['in_dim'], "num_layers": config['executor']['head'],
                "hidden_size": config['executor']['out_dim'] * config['executor']['head']}
    net = rnnModel(paramter).to(device)
    lr = 0.001
    cost = torch.nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    epoch_loss = 0
    epoch_num = 1
    for epoch in range(epoch_num):  # 每个epoch循环
        logging.info(f'Epoch {epoch}/{epoch_num}')
        for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
            if 'cuda' in str(device):
                node = node.to(device=device)
            tid1 = np.array(tid1)
            output, hn = net(node)

            tid1_vec = output[tid1].to(device=device)
            tid2_vec = output[tid2].to(device=device)

            label = torch.tensor([i for i in range(len(tid1))]).to(device=device)
            sim1 = torch.matmul(tid1_vec, tid2_vec.T)
            loss = cost(sim1, label)
            epoch_loss += loss.data.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(f"loss = {loss.data.item()}")
        epoch_loss = epoch_loss / len(data_loader)
        logging.info(f"epoch loss:{epoch_loss}")
        torch.save(net.state_dict(), f'{log_path}rnn_{name.split("_")[-1]}.pth')

    for k, v in test_data.items():  # file_name: tid
        logging.info(f"{k}...")
        print(f"{k}...")
        graph_data = GraphLoader(config['path'], v)
        data_loader = DataLoader(dataset=graph_data, batch_size=16, num_workers=4,
                                 collate_fn=rnn_coll, persistent_workers=True, shuffle=True)
        embedding_1 = []
        embedding_2 = []
        for node, tid1, tid2 in data_loader:  # 每个批次循环
            if 'cuda' in str(device):
                node = node.to(device=device)
            output, hn = net(node)
            vec1 = output[tid1]
            vec2 = output[tid2]
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



def pipeline():

    # signature()
    bl_rnn()
    # bl_gnn()
    # bl_transformer()


if __name__ == "__main__":
    name = "STEL_taxi"  # "STEL_ais", "STEL_taxi"
    config = parse_config(name)
    log_path = "./libTrajectory/logs/STEL/baseline/"
    logging.basicConfig(filename=f'{log_path}rnn_{name.split("_")[-1]}.log',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    pipeline()
