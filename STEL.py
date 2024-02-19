import logging
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


log_path = "./libTrajectory/logs/STEL/"
data_path = "./libTrajectory/dataset/AIS/"


def pipeline():
    logging.basicConfig(filename=f'{log_path}preprocess.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    preprocessor = Preprocessor(f"{data_path}test4K.csv",
                                {"test1": f"{data_path}test1K.csv", "test2": f"{data_path}test3K.csv"})
    # preprocessor.run()  # 注释后get方法将调用已经预处理好的数据，
    # tid : 轨迹标识，tid1与tid2相同则为正样本，否则为负样本
    train_data, test_data, vector = preprocessor.get()
    executor = Executor()
    executor.train(train_data, vector)


if __name__ == "__main__":
    pipeline()
