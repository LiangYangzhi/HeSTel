import logging

from libTrajectory.preprocessing.STEL.pretrain import Pretrain


def pipeline():
    log_path = "./libTrajectory/logs/STEL/"
    logging.basicConfig(filename=f'{log_path}multiA.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    route = "./libTrajectory/dataset/AIS/"
    data_path = f"{route}multiA.csv"  # multiA  sample6K
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    pretrain = Pretrain(data_path)
    pretrain.get(method="run")
    pretrain.eval(test_path)


if __name__ == "__main__":
    pipeline()
