import logging
import os

from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor, save_graph
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = f"./libTrajectory/logs/STEL/sim/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}326-1.log', format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    route = "./libTrajectory/dataset/myAIS/"
    logging.info("enhance sample add spaceid and timeid")

    data_path = f"{route}multiA.csv"
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    train_tid, test_tid, stid_counts, enhance_ns = Preprocessor(data_path, test_path).get(method='load')

    executor = Executor(route, stid_counts, log_path, in_dim=32, cuda=0)
    executor.train(train_tid, enhance_ns, test_tid, epoch_num=1, batch_size=128, num_workers=16)


if __name__ == "__main__":
    pipeline()
