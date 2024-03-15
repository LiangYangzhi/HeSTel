import logging

from libTrajectory.preprocessing.STEL.graphGenerator import GraphPreprocessor
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = "./libTrajectory/logs/STEL/"
    logging.basicConfig(filename=f'{log_path}train3_13.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    route = "./libTrajectory/dataset/AIS/"
    data_path = f"{route}multiA.csv"  # multiA  sample6K
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    train_data, test_data, stid_counts = Preprocessor(data_path, test_path).get(method='load')
    executor = Executor(stid_counts)
    executor.train(train_data, test_data)  # test_data传入，则epoch_num每次结束都进行infer
    # executor.infer(test_data)


if __name__ == "__main__":
    pipeline()
