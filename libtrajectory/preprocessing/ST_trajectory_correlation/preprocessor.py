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

        print(f"--{self.config['data1']['subset']}, city: {self.config['data1']['city']}, \n"
              f"--start_time: {self.config['data1']['start_time']}, end_time: {self.config['data1']['end_time']}")
        self.data1 = data_loader.reading(**self.config['data1'])
        self.col1 = self.config['data1']['columns']

        print(f"--{self.config['data2']['subset']}, city: {self.config['data2']['city']}, \n"
              f"--start_time: {self.config['data2']['start_time']}, end_time: {self.config['data2']['end_time']}")
        self.data2 = data_loader.reading(**self.config['data2'])
        self.col2 = self.config['data2']['columns']

        print(f"--load label: {self.config['label']['file']}")
        self.label = data_loader.reading_label(**self.config['label'])
        self.label["label"] = 1
        self.col_label = [self.col1["user"], self.col2["user"], "label"]
        self.label = self.label[self.col_label]
        self.label = self.label.sample(50000, random_state=34078)

    def data_cleaning(self):
        self.label: pd.DataFrame
        self.label.dropna(inplace=True)
        self.label[self.col1["user"]] = self.label[self.col1["user"]].astype('str')
        self.label[self.col2["user"]] = self.label[self.col2["user"]].astype('str')

        self.data1: pd.DataFrame
        self.data1.dropna(inplace=True)
        self.data1[self.col1["user"]] = self.data1[self.col1["user"]].astype('str')
        self.data1[self.col1["user"]] = self.data1[self.col1["user"]].astype('category')
        self.data1[self.col1["longitude"]] = self.data1[self.col1["longitude"]].astype('category')
        self.data1[self.col1["latitude"]] = self.data1[self.col1["latitude"]].astype('category')
        self.data1[self.col1["device"]] = self.data1[self.col1["device"]].astype('category')
        # self.data1 keep only label data
        label_user1 = self.label[self.col_label[0]].unique().tolist()
        self.data1 = self.data1[self.data1[self.col1['user']].isin(label_user1)]

        self.data2: pd.DataFrame
        self.data2.dropna(inplace=True)
        self.data2[self.col2["user"]] = self.data2[self.col2["user"]].astype('str')
        self.data2[self.col2["user"]] = self.data2[self.col2["user"]].astype('category')
        self.data2[self.col2["longitude"]] = self.data2[self.col2["longitude"]].astype('category')
        self.data2[self.col2["latitude"]] = self.data2[self.col2["latitude"]].astype('category')
        self.data2[self.col2["device"]] = self.data2[self.col2["device"]].astype('category')

    def data_augmentation(self):
        print("--data segmentation")  # Todo
        data_augmentation = DataAugmentation()
        self.segment, self.col_segment = data_augmentation.non_segmentation(
            data=self.data1, col=[self.col1['user'], self.col1['time']])
        # self.segment = self.segment.sample(30000, random_state=89734)

    def feature_engineering(self):
        print("--determining candidate pairs")
        pairs_class = DeterminingCandidatePairs(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2)
        self.pairs, self.col_pairs = pairs_class.sti_place_num_index(
            segment=self.segment, col_segment=self.col_segment, **self.config['pairs'])
        self.pairs.to_csv("./pairs.csv", index=False)

        print("--feature extraction")
        feature = FeatureExtraction(data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2)
        self.feature, self.col_feature = feature.sequence_index(
            pairs=self.pairs, col_pairs=self.col_pairs, ** self.config['feature'])
        self.feature.to_csv("./feature.csv", index=False)

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
        info = self.config['splitting']['info']
        divide_train = self.config['splitting']['divide_train']
        divide_test = self.config['splitting']['divide_train']

        if info == "ratio":
            frac = divide_train / (divide_train + divide_test)
            train_label = self.label.sample(frac=frac, random_state=42)
            test_label = self.label.drop(train_label.index)
            label_col = self.col_label[2]
            feature_col = [c for c in self.feature.columns if c not in self.col_pairs.values()]

            train = pd.merge(self.feature, train_label, how="left")
            train[label_col].fillna(0, inplace=True)
            train['temp_unique'] = train.apply(
                lambda row: str(row[self.col_feature['user1']]) + str(row[self.col_feature['segment']]), axis=1)
            positives = train[train[label_col] == 1]['temp_unique'].unique().tolist()
            train = train[train['temp_unique'].isin(positives)]
            train.drop(columns=['temp_unique'], inplace=True)
            y_train = train[label_col].to_list()

            drop_col = [c for c in train.columns if c not in feature_col]
            X_train = train.drop(columns=drop_col)

            test = pd.merge(self.feature, test_label, how="left")
            test[label_col].fillna(0, inplace=True)
            test['temp_unique'] = test.apply(
                lambda row: str(row[self.col_feature['user1']]) + str(row[self.col_feature['segment']]), axis=1)
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
