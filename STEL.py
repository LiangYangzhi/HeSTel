import logging
import os

from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    cuda = 1
    log_path = f"./libTrajectory/logs/STEL/loss/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}loss6.log', format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    if cuda == 0:
        route = "/run/dataset/AIS/new_data/"
    else:
        route = "./libTrajectory/dataset/AIS/new_data/"
    logging.info("enhance sample add spaceid and timeid")

    data_path = f"{route}multiA.csv"
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    train_tid, test_tid, stid_counts, enhance_ns = Preprocessor(data_path, test_path).get(method='load')

    executor = Executor(route, stid_counts, log_path, cuda=cuda)
    executor.train(train_tid, enhance_ns, test_tid, epoch_num=3, batch_size=256, num_workers=18)
    # executor.infer(test_tid)


if __name__ == "__main__":
    pipeline()
