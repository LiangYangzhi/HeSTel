import logging
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = "./libTrajectory/logs/STEL/"
    logging.basicConfig(filename=f'{log_path}train319.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    route = "./libTrajectory/dataset/AIS/"
    data_path = f"{route}multiA.csv"  # multiA  sample6K
    test_path = {"test1": f"{route}test1K.csv", "test2": f"{route}test3K.csv"}
    train_tid, test_tid, stid_counts, enhance_ns = Preprocessor(data_path, test_path).get(method='load')

    executor = Executor(data_path, stid_counts)
    executor.train(train_tid, enhance_ns, test_tid)  # test_data传入，则epoch_num每次结束都进行infer
    # executor.infer(test_tid)


if __name__ == "__main__":
    pipeline()
