import pandas as pd

from libtrajectory.dataset.dataset_load import DataLoader
from libtrajectory.preprocessing.ST_trajectory_correlation.DataAugmentation import DataAugmentation
from libtrajectory.preprocessing.ST_trajectory_correlation.determining_candidate_pairs import DeterminingCandidatePairs
from libtrajectory.preprocessing.ST_trajectory_correlation.feature_extraction import FeatureExtraction
from libtrajectory.preprocessing.abstract_preprocessor import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    def __init__(self, config):
        self.col_feature = None
        self.feature = None
        self.col_pairs = None
        self.pairs = None
        self.segment = None
        self.col_segment = None
        self.label = None
        self.col_label = None
        self.data2 = None
        self.col2 = None
        self.data1 = None
        self.col1 = None
        self.config = config

    def data_loading(self):
        data_loader = DataLoader()

        self.data1 = data_loader.reading(**self.config['data1'])
        self.col1 = self.config['data1']['columns']

        self.data2 = data_loader.reading(**self.config['data2'])
        self.col2 = self.config['data2']['columns']

        self.label = data_loader.reading(**self.config['label'])
        self.label["label"] = 1
        self.col_label = [self.col1["user"], self.col2["user"], "label"]
        self.label = self.label[self.col_label]

    def data_cleaning(self):
        self.data1: pd.DataFrame
        self.data1.dropna(inplace=True)
        self.data2: pd.DataFrame
        self.data2.dropna(inplace=True)
        self.label: pd.DataFrame
        self.label.dropna(inplace=True)

    def data_augmentation(self):
        print("--data segmentation")  # Todo
        data_augmentation = DataAugmentation()
        self.segment, self.col_segment = data_augmentation.data_segmentation(
            data=self.data1, col=[self.col1['user'], self.col1['time']], **self.config['segment'])

    def _index(self):
        index1 = [self.col1['user'], self.col1['time'], self.col1['device']]  # 决定取值的顺序，index通过位置获取values
        self.data1.set_index(index1, drop=True, inplace=True)
        index2 = [self.col2['user'], self.col2['time'], self.col2['device']]
        self.data2.set_index(index2, drop=True, inplace=True)

    def feature_engineering(self):
        # self._index()  # pairs and feature 使用index后缀的方法
        print("--determining candidate pairs")
        determining_candidate_pairs = DeterminingCandidatePairs(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2)
        self.pairs, self.col_pairs = determining_candidate_pairs.sti_place_num(
            segment=self.segment, col_segment=self.col_segment, **self.config['pairs'])

        print("--feature extraction")
        feature = FeatureExtraction(data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2)
        self.feature, self.col_feature = feature.sequence(
            pairs=self.pairs, col_pairs=self.col_pairs, ** self.config['feature'])

    def splitting(self):
        """
        info: str
            "time" or "ratio"
        divide_train:
            if info == "time", train = [start_time, end_time), example: start_time = time stamp
            if info == "ratio", train = int (ratio=train/(train+test))
        divide_test:
            if info == "time", test = [start_time, end_time)
            if info == "ratio", test = int (ratio=train/(train+test))
        """
        self.feature.to_csv("feature.csv")
        info = self.config['splitting']['info']
        divide_train = self.config['splitting']['divide_train']
        divide_test = self.config['splitting']['divide_train']

        if info == "ratio":
            frac = divide_train / (divide_train + divide_test)
            train_label = self.label.sample(frac=frac, random_state=42)
            test_label = self.label.drop(train_label.index)
            label_col = self.col_label[2]
            feature_col = [c for c in self.feature.columns if c not in self.col_pairs]

            train = pd.merge(self.feature, train_label, how="left")
            train[label_col].fillna(0, inplace=True)
            train['temp_unique'] = train.apply(
                lambda row: str(row[self.col_feature[0]]) + str(row[self.col_feature[2]]), axis=1)
            positives = train[train[label_col] == 1]['temp_unique'].unique().tolist()
            train = train[train['temp_unique'].isin(positives)]
            train.drop(columns=['temp_unique'], inplace=True)
            y_train = train[label_col].to_list()

            drop_col = [c for c in train.columns if c not in feature_col]
            X_train = train.drop(columns=drop_col)

            test = pd.merge(self.feature, test_label, how="left")
            test[label_col].fillna(0, inplace=True)
            test['temp_unique'] = test.apply(
                lambda row: str(row[self.col_feature[0]]) + str(row[self.col_feature[2]]), axis=1)
            positives = test[test[label_col] == 1]['temp_unique'].unique().tolist()
            test = test[test['temp_unique'].isin(positives)]
            test.drop(columns=['temp_unique'], inplace=True)
            y_test = test[label_col].to_list()

            drop_col = [c for c in test.columns if c not in feature_col]
            index = test[drop_col]
            index.reset_index(drop=True, inplace=True)
            X_test = test.drop(columns=drop_col)

            return X_train, y_train, X_test, y_test, index

        if info == "time":
            pass

    def normalization(self):
        pass  # Todo
