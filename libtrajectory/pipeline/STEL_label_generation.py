"""
Pipeline:
Data preparation
Label generation
"""
import datetime

from libtrajectory.logs.STEL_label_generation.logs import Logs
from libtrajectory.preprocessing.STEL_label_generation.preprocessor import Preprocessor
from libtrajectory.utils.time_utils import parse_time


def pipeline(config):
    logs = Logs()
    pre_config = config['preprocessing']
    start_time = parse_time(pre_config['time']['start_time'])
    end_time = parse_time(pre_config['time']['end_time'])  #
    while True:
        loop_start_time = start_time
        if start_time + datetime.timedelta(days=2) > end_time:
            start_time = end_time
        else:
            start_time = start_time + datetime.timedelta(days=2)
        pre_config['time']['start_time'] = loop_start_time
        pre_config['time']['end_time'] = start_time
        print(loop_start_time, start_time)

        preprocessor = Preprocessor(pre_config)
        preprocessor.data_loading(sample=config['sample'])
        preprocessor.data_cleaning()  # Todo
        preprocessor.data_augmentation()
        preprocessor.feature_engineering()
        feature, feature_col = preprocessor.get_data()
        logs.save_feature(feature, pre_config['time']['start_time'], pre_config['time']['end_time'])

        if start_time == end_time:
            break
    logs.save_params()
