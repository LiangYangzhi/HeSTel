# import logging
#
# from libTrajectory.preprocessing.STEL.negative_sample import NegativeSample
# from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
# from libTrajectory.executor.STEL import Executor
#
#
# log_path = "./libTrajectory/logs/STEL/"
# data_path = "./libTrajectory/dataset/AIS/"
#
#
# def pipeline():
#     logging.basicConfig(filename=f'{log_path}NegativeSample.log', format='%(asctime)s - %(levelname)s - %(message)s',
#                         level=logging.INFO)
#     # preprocessor = Preprocessor(f"{data_path}multiA.csv",
#     #                             {"test1": f"{data_path}test1K.csv", "test2": f"{data_path}test3K.csv"})
#     sample = NegativeSample(f"{data_path}multiA.csv")
#     sample.run()
#     data1, data2 = sample.get()
#     # preprocessor.run()  # 注释后get方法将调用已经预处理好的数据，
#     # tid : 轨迹标识，tid1与tid2相同则为正样本，否则为负样本.
#     # train_data, test_data, st_vec, stid_counts = preprocessor.get()
#     # executor = Executor(st_vec, stid_counts)
#     # executor.train(train_data)
#     # executor.infer(test_data)
#
#
# if __name__ == "__main__":
#     pipeline()


import sys

my_variable = [1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 11, 23, 24]
print(sys.getsizeof(my_variable))

