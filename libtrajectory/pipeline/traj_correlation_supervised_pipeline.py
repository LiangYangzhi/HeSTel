from datetime import datetime

from libtrajectory.executor.correlation_executor import LightGBMExecutor
from libtrajectory.preprocessing.ST_trajectory_correlation.preprocessor import Preprocessor
from libtrajectory.evaluator.precision import evaluation

"""
Pipeline:
Data preparation
Model definition
Model training
Model prediction
Model evaluation 
Model saving and loading
Result analysis and visualization
"""


def pipeline(config):
    # train
    print("\ntrain dataset")
    preprocessor = Preprocessor(config['preprocessing'], info="train")
    preprocessor.data_loading()
    if config["test"]:
        preprocessor.label_sample()
    preprocessor.data_cleaning()  # Todo
    preprocessor.data_augmentation()
    preprocessor.feature_engineering()
    X_train, y_train = preprocessor.get_data()

    # test
    print("\ntest dataset")
    preprocessor = Preprocessor(config['preprocessing'], info="test")
    preprocessor.data_loading()
    if config["test"]:
        preprocessor.label_sample()
    preprocessor.data_cleaning()  # Todo
    preprocessor.data_augmentation()
    preprocessor.feature_engineering()
    X_test, y_test, index = preprocessor.get_data()  # index 是为了后续evaluate服务

    executor = LightGBMExecutor(X_train=X_train, y_train=y_train, X_test=X_test, config=config['model'])
    pred = executor.run()

    evaluation(X_test, y_test, index, pred, config)  # Todo 完善评价标准化

    if config.get("save", None):
        pass  # Todo 完善 trained_model模块 (model file, params file, evaluation file)
