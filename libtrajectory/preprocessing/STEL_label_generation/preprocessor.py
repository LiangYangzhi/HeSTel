import pandas as pd
from datetime import datetime

from pandas import DataFrame

from libtrajectory.preprocessing.STEL_label_generation.data_augmentation import DataAugmentation
from libtrajectory.preprocessing.STEL_label_generation.dataset_load import DataLoader
from libtrajectory.preprocessing.STEL_label_generation.determining_candidate_pairs import \
    DeterminingCandidatePairs
from libtrajectory.preprocessing.abstract_preprocessor import AbstractPreprocessor
from libtrajectory.utils.coordinate import calculate_device_distance


class Preprocessor(AbstractPreprocessor):

    def __init__(self, config):
        self.col_feature = None
        self.feature = None
        self.col_segment = None
        self.segment = None
        self.device_distance = None
        self.col2 = None
        self.data2 = None
        self.col1 = None
        self.data1 = None
        self.config = config

    def data_loading(self, sample):
        print("data loading")
        start = datetime.now()
        data_loader = DataLoader(**self.config['time'])
        self.data1 = data_loader.get_data(**self.config['data1'])
        self.col1 = self.config['data1']['columns']
        if sample:
            self.data1 = self.data1.sample(sample, random_state=23742)
        print(f"data1 number : {self.data1.shape}")

        self.data2 = data_loader.get_data(**self.config['data2'])
        self.col2 = self.config['data2']['columns']
        print(f"data2 number : {self.data2.shape}")

        self.device_distance = calculate_device_distance(self.data1, self.col1, self.data2, self.col2)
        device1 = self.device_distance[self.col1['device']].unique().tolist()
        self.data1 = self.data1[self.data1[self.col1['device']].isin(device1)]
        print(f"data1 number after filter device1 beyond device2 coverage: {self.data1.shape}")
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def data_cleaning(self):
        pass  # Todo

    def data_augmentation(self):
        print("data augmentation")
        start = datetime.now()
        augmenter = DataAugmentation(
            data=self.data1[[self.col1['user'], self.col1['time']]],
            col={"user": self.col1['user'], "time": self.col1['time']})
        self.segment, self.col_segment = augmenter.no_augmentation()
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def feature_engineering(self):
        print("feature engineering")
        start = datetime.now()

        config = self.config['pairs']
        pairs_class = DeterminingCandidatePairs(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2,
            batch_size=self.config['batch_size'], front_time=config['front_time'], back_time=config['back_time'],
            device_distance=self.device_distance)

        self.feature, self.col_feature = pairs_class.sti_place_num(
            segment=self.segment, col_segment=self.col_segment,
            place_top=config['place_top'], num_top=config['num_top']
        )
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def get_data(self):
        return self.feature, self.col_feature
