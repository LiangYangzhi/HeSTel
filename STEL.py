import logging
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


log_path = "./libTrajectory/logs/STEL/"
data_path = "./libTrajectory/dataset/AIS/"


def pipeline():
    logging.basicConfig(filename=f'{log_path}STEL.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    preprocessor = Preprocessor(f"{data_path}multiA.csv",
                                {"test1": f"{data_path}test1K.csv", "test2": f"{data_path}test3K.csv"})
    preprocessor.run()  # 注释后get方法将调用已经预处理好的数据，
    # tid : 轨迹标识，tid1与tid2相同则为正样本，否则为负样本.
    train_data, test_data, ts_vec, st_vec, tsid_counts, stid_counts = preprocessor.get()
    executor = Executor(tsid_counts, stid_counts)
    executor.train(train_data, ts_vec, st_vec)


if __name__ == "__main__":
    pipeline()
