"""
Pipeline:
Data preparation
Label generation
"""
import pandas as pd

from libtrajectory.logs.STEL_STIntersection.logs import Logs
from libtrajectory.preprocessing.STEL_STIntersection.preprocessor import Preprocessor


def pipeline(config):
    logs = Logs()

    preprocessor = Preprocessor(config['preprocessing'])
    preprocessor.data_loading(sample=config['sample'])
    preprocessor.data_cleaning()  # Todo
    preprocessor.data_augmentation()
    preprocessor.feature_engineering()
    feature, feature_col, label = preprocessor.get_data()
    data = pd.merge(feature, label, how='left')
    data['label'].fillna(0, inplace=True)
    t_config = config['preprocessing']['time']
    logs.save_feature(data, t_config['start_time'], t_config['end_time'])

    # feature: device、device_set_num
    print("evaluating...")
    data['rank'] = data.groupby(feature_col['user1'])['device_set_num'].rank(ascending=False, method='first')
    label_num = data.query("label == 1").label.tolist().__len__()
    top1 = len(data.query("(rank == 1) & (label == 1)"))
    print(f"top1: {top1} / {label_num} = {top1/label_num}")
    top5 = len(data.query("(rank <= 5) & (label == 1)"))
    print(f"top5: {top5} / {label_num} = {top5 / label_num}")

    logs.save_params()
