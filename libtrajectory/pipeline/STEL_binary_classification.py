from datetime import datetime

from libtrajectory.executor.STEL_executor import LightGBMExecutor
from libtrajectory.logs.STEL_binary_classification.logs import Logs
from libtrajectory.preprocessing.STEL_binary_classification.preprocessor import Preprocessor
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
    logs = Logs()
    # train
    print("\ntrain dataset")
    preprocessor = Preprocessor(config['preprocessing'], info="train")
    preprocessor.data_loading(config["sample"])
    preprocessor.data_cleaning()  # Todo
    preprocessor.data_augmentation()
    feature = preprocessor.feature_engineering()
    start_time = config['preprocessing']['train']['start_time']
    end_time = config['preprocessing']['train']['end_time']
    logs.save_feature(feature, start_time, end_time)
    X_train, y_train = preprocessor.get_data()

    # test
    print("\ntest dataset")
    preprocessor = Preprocessor(config['preprocessing'], info="test")
    preprocessor.data_loading(config["sample"])
    preprocessor.data_cleaning()  # Todo
    preprocessor.data_augmentation()
    feature = preprocessor.feature_engineering()
    start_time = config['preprocessing']['test']['start_time']
    end_time = config['preprocessing']['test']['end_time']
    logs.save_feature(feature, start_time, end_time)
    X_test, y_test, index = preprocessor.get_data()

    executor = LightGBMExecutor(X_train=X_train, y_train=y_train, X_test=X_test, config=config['model'])
    pred = executor.run()
    executor.save_model(f"{logs.get_path()}/model.txt")
    evaluation(X_test, y_test, index, pred, config)  # Todo 完善评价标准化

    logs.save_params()
