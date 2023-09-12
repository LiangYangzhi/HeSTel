import pandas as pd

from libtrajectory.config.config_parser import ConfigParser
from libtrajectory.executor.correlation_executor import LightGBMExecutor
from libtrajectory.model.lightgbm_model import LightgbmModel
from libtrajectory.preprocessing.ST_trajectory_correlation.preprocessor import Preprocessor
from libtrajectory.evaluator.precision import precision

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
    preprocessor = Preprocessor(config['preprocessing'])
    print("data loading")
    preprocessor.data_loading()
    print("data cleaning")
    preprocessor.data_cleaning()  # Todo
    print("triggering mechanism")
    preprocessor.triggering_mechanism()
    print("feature engineering")
    preprocessor.feature_engineering()
    print("data splitting")
    X_train, y_train, X_test, y_test, index = preprocessor.splitting()

    model = LightgbmModel()
    model.set_params(**config['model']['params'])

    executor = LightGBMExecutor(model)
    executor.train(X_train, y_train)
    pred = executor.predict(X_test)
    pred = pred.reshape((X_test.shape[0], 1))
    pred = pd.DataFrame(data=pred, columns=['probably'])
    data = pd.merge(index, pred, how="outer", left_index=True, right_index=True)

    group_name = [config['preprocessing']['data1']['columns']['user'], 'task']
    df_sort = data.sort_values(by='probably', ascending=False).groupby(group_name)
    # top1 precision
    df1: pd.DataFrame = df_sort.head(1)
    df1.insert(0, column='pred', value=1)
    precision1 = precision(df1, ["label", "pred"])
    print(precision1)

    # top5 precision
    df5: pd.DataFrame = df_sort.head(5)
    df5.insert(0, column='pred', value=1)
    precision5 = precision(df1, ["label", "pred"])
    print(precision5)

