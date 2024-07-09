import logging
import os
import random

import pandas as pd

from libTrajectory.config.config_parser import parse_config
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    path = config['path']
    test_file = {"test1": "test1K.csv", "test2": "test3K.csv"}
    log_path = f"./libTrajectory/logs/STEL/taxi/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}pre1.log', format='%(asctime)s - %(message)s', level=logging.INFO)

    train_tid, test_tid, enhance_tid = Preprocessor(path, test_file, config['preprocessing']).get(method='run')
    executor = Executor(path, log_path, config['executor'])
    executor.train(train_tid, enhance_tid, test_tid)


if __name__ == "__main__":
    config = parse_config("/STEL_taxi")  # STEL_ais STEL_taxi
    pipeline()
