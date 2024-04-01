import logging
import os

from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor, save_graph
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = f"./libTrajectory/logs/STEL/ais/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}process.log', format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    path = "./libTrajectory/dataset/ais/"
    logging.info("enhance sample add spaceid and timeid")

    test_file = {"test1": "test1K.csv", "test2": "test3K.csv"}
    # Preprocessor(path, test_file).get(method='run')
    save_graph(path, test_file)
    train_tid, test_tid, stid_counts, enhance_ns = Preprocessor(path, test_file).get(method='load')

    executor = Executor(path, stid_counts, log_path, in_dim=34, cuda=0)
    executor.train(train_tid, enhance_ns, test_tid, epoch_num=1, batch_size=128, num_workers=16)


if __name__ == "__main__":
    pipeline()
