import logging
import os

from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = f"./libTrajectory/logs/STEL/Transformer/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}test.log', format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    path = "./libTrajectory/dataset/ais/"
    logging.info("enhance sample add spaceid and timeid")

    test_file = {"test1": "test1K.csv", "test2": "test3K.csv"}
    train_tid, test_tid, enhance_ns = Preprocessor(path, test_file).get(method='load')

    executor = Executor(path, log_path, in_dim=34, cuda=0)
    executor.train(train_tid, enhance_ns, test_tid, epoch_num=1, batch_size=128, num_workers=12)


if __name__ == "__main__":
    pipeline()
