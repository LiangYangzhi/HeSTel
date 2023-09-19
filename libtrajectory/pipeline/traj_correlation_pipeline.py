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
    start = datetime.now()
    preprocessor = Preprocessor(config['preprocessing'])
    print("data loading")
    preprocessor.data_loading()
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    print("data cleaning")
    start = datetime.now()
    preprocessor.data_cleaning()  # Todo
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    print("data augmentation")
    start = datetime.now()
    preprocessor.data_augmentation()
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    print("feature engineering")
    start = datetime.now()
    preprocessor.feature_engineering()
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    print("data splitting")
    start = datetime.now()
    X_train, y_train, X_test, y_test, index = preprocessor.splitting()  # index 是为了后续evaluate服务
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    print("model")
    start = datetime.now()
    executor = LightGBMExecutor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, config=config['model'])
    pred = executor.run()
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    # Todo 完善评价标准化
    print("evaluation")
    start = datetime.now()
    evaluation(X_test, index, pred, config)
    print(f'Running time: {datetime.now() - start} Seconds', '\n')

    if config.get("save", None):
        pass  # Todo 完善 trained_model模块 (model file, params file, evaluation file)
