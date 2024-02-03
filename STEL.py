import logging
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.preprocessing.STEL.graphLoader import IdDataset, GraphDataset
from libTrajectory.executor.STEL import Executor


def pipeline():
    logging.basicConfig(filename='./libTrajectory/logs/STEL/example.log',
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    preprocessor = Preprocessor("./libTrajectory/dataset/AIS/test10.csv")
    # tid : 用户标识，tid1与tid2相同则为正样本，否则为负样本
    data1, ts_vec, data2, st_vec = preprocessor.get()
    executor = Executor()
    executor.train(data1, ts_vec, data2, st_vec)
