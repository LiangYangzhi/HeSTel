import logging
import pickle
import random

from torch.utils.data import DataLoader
from tqdm import tqdm

from libTrajectory.config.config_parser import parse_config
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
    train_tid, test_tid, enhance_tid = NetPre(config).get(method='load')
    graph_data = GraphLoader(config['path'], train_tid)
    data_loader = DataLoader(dataset=graph_data, batch_size=2, num_workers=8,
                             collate_fn=rnn_coll, persistent_workers=True, shuffle=True)
    epoch_num = 1
    for epoch in range(epoch_num):  # 每个epoch循环
        logging.info(f'Epoch {epoch}/{epoch_num}')
        for node, tid1, tid2 in tqdm(data_loader):  # 每个批次循环
            print(node)
            print(tid1)
            print(tid2)

    # rnn = torch.nn.RNN(10, 20, 2)
    #
    # input = torch.randn(5, 3, 10)
    # print(input.size())
    # h0 = torch.randn(2, 3, 20)
    # print(h0.size())
    # output, hn = rnn(input, h0)
    # print(output.size())
    # print(hn.size())


def pipeline():
    log_path = "./libTrajectory/logs/STEL/baseline/"
    logging.basicConfig(filename=f'{log_path}test_rnn_{name.split("_")[-1]}.log',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    # signature()
    bl_rnn()
    # bl_gnn()
    # bl_transformer()


if __name__ == "__main__":
    name = "STEL_taxi"  # "STEL_ais", "STEL_taxi"
    config = parse_config(name)
    pipeline()
