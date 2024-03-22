import logging
import os

from libTrajectory.preprocessing.STEL.pretrain import Pretrain


def pipeline():

    log_path = f"./libTrajectory/logs/STEL/pretrain/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=f'{log_path}pretrain32.log', format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    route = "./libTrajectory/dataset/AIS/new_data/"
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    logging.info("enhance sample add spaceid and timeid")

    pretrian = Pretrain(route)
    pretrian.run()
    pretrian.eval(test_path)


if __name__ == "__main__":
    pipeline()
