import pandas as pd
from datetime import datetime

from libtrajectory.preprocessing.STEL_binary_classification.dataset_load import DataLoader
from libtrajectory.preprocessing.STEL_binary_classification.data_augmentation import DataAugmentation
from libtrajectory.preprocessing.STEL_binary_classification.determining_candidate_pairs import DeterminingCandidatePairs
from libtrajectory.preprocessing.STEL_binary_classification.feature_extraction import FeatureExtraction
from libtrajectory.preprocessing.abstract_preprocessor import AbstractPreprocessor
from libtrajectory.utils.coordinate import calculate_device_distance


class Preprocessor(AbstractPreprocessor):
    def __init__(self, config, info):
        self.device_distance = None
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
        self.info = info  # train or test

        if self.info == "train":
            self.config["start_time"] = self.config["train"]["start_time"]
            self.config["end_time"] = self.config["train"]["end_time"]
        elif self.info == "test":  # test
            self.config["start_time"] = self.config["test"]["start_time"]
            self.config["end_time"] = self.config["test"]["end_time"]
        else:
            self.config["start_time"] = self.config["start_time"]
            self.config["end_time"] = self.config["end_time"]

    def data_loading(self, sample):
        print("data loading")
        start = datetime.now()
        data_loader = DataLoader()

        self.data1 = data_loader.get_data(
            start_time=self.config["start_time"], end_time=self.config["end_time"],
            **self.config['data1'])
        self.col1 = self.config['data1']['columns']
        print(f"data1 number : {self.data1.shape}")

        self.data2 = data_loader.get_data(
            start_time=self.config["start_time"], end_time=self.config["end_time"],
            **self.config['data2'])
        self.col2 = self.config['data2']['columns']
        print(f"data2 number : {self.data2.shape}")

        self.device_distance = calculate_device_distance(self.data1, self.col1, self.data2, self.col2)
        device1 = self.device_distance[self.col1['device']].unique().tolist()
        self.data1 = self.data1[self.data1[self.col1['device']].isin(device1)]
        print(f"data1 number after filter device1 beyond device2 coverage: {self.data1.shape}")

        self.label = data_loader.get_data(**self.config['label'])
        self.label["label"] = 1
        self.col_label = [self.col1["user"], self.col2["user"], "label"]
        print(self.label.info())
        print(sample)
        if sample:
            self.label = self.label.sample(sample, random_state=34078)
        print(f"label number : {self.label.shape}")
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def data_cleaning(self):
        print("data cleaning")
        start = datetime.now()

        # self.data1 keep only label data
        label_user1 = self.label[self.col_label[0]].unique().tolist()
        self.data1 = self.data1[self.data1[self.col1['user']].isin(label_user1)]
        print(f"data1 number after filter non label user: {self.data1.shape}")

        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def data_augmentation(self):
        print("data augmentation")
        start = datetime.now()
        augmenter = DataAugmentation(
            data=self.data1[[self.col1['user'], self.col1['time']]],
            col={"user": self.col1['user'], "time": self.col1['time']})
        self.segment, self.col_segment = augmenter.max_length()
        print(self.segment.info())
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def feature_engineering(self):
        print("feature engineering")
        start = datetime.now()

        print("--determining candidate pairs")
        config = self.config['pairs']
        pairs_class = DeterminingCandidatePairs(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2,
            batch_size=self.config['batch_size'], front_time=config['front_time'], back_time=config['back_time'],
            device_distance=self.device_distance)

        self.pairs, self.col_pairs = pairs_class.sti_place_num(
            segment=self.segment, col_segment=self.col_segment,
            place_top=config['place_top'], num_top=config['num_top']
        )
        print(self.pairs.info())
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

        if self.info == "train":
            self._drop_ns_group()

        print("--feature extraction")
        extractor = FeatureExtraction(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2,
            batch_size=self.config['batch_size'], **self.config['feature'])
        self.feature, self.col_feature = extractor.sequence(
            pairs=self.pairs, col_pairs=self.col_pairs)
        print(self.feature.info())
        print(f'Running time: {datetime.now() - start} Seconds', '\n')
        return self.feature

    def get_data(self, method=None):
        self.index = [self.col1["user"], self.col2['user'], "segment"]  # todo 为了后续对测试进行评价

        if method == "all":
            X_train, y_train, X_test, y_test, index = self._data_splitting(0.8)
            return X_train, y_train, X_test, y_test, index

        elif self.info == "train":
            train = pd.merge(self.feature, self.label, how="left")
            train["label"].fillna(0, inplace=True)
            y_train = train["label"].to_list()
            X_train = train[self.col_feature["feature"]]
            return X_train, y_train

        elif self.info == "test":
            test = pd.merge(self.feature, self.label, how="left")
            test["label"].fillna(0, inplace=True)
            y_test = test["label"].to_list()
            X_test = test[self.col_feature["feature"]]
            index = test[self.index]
            index.reset_index(drop=True, inplace=True)
            return X_test, y_test, index

    def normalization(self):
        pass  # Todo

    def _data_splitting(self, train_ratio):
        train_label = self.label.sample(frac=train_ratio, random_state=4234)
        test_label = self.label.drop(train_label.index)

        # train
        train = pd.merge(self.feature, train_label, how="left")
        train["label"].fillna(0, inplace=True)

        # 去除无正样本的组
        train['temp_unique'] = train.apply(
            lambda row: str(row[self.col_feature['user1']]) + str(row[self.col_feature['segment']]), axis=1)
        positives = train[train["label"] == 1]['temp_unique'].unique().tolist()
        train = train[train['temp_unique'].isin(positives)]
        train.drop(columns=['temp_unique'], inplace=True)

        y_train = train[self.col_label["label"]].to_list()
        X_train = train[self.col_feature["feature"]]

        # test
        test = pd.merge(self.feature, test_label, how="left")
        test["label"].fillna(0, inplace=True)

        # 去除无正样本的组
        test['temp_unique'] = test.apply(
            lambda row: str(row[self.col_feature['user1']]) + str(row[self.col_feature['segment']]), axis=1)
        positives = test[test["label"] == 1]['temp_unique'].unique().tolist()
        test = test[test['temp_unique'].isin(positives)]
        test.drop(columns=['temp_unique'], inplace=True)

        y_test = test[self.col_label[-1]].to_list()
        index = test[self.index]
        index.reset_index(drop=True, inplace=True)
        X_test = test[self.col_feature["feature"]]

        return X_train, y_train, X_test, y_test, index

    def _drop_ns_group(self):
        print("---Remove groups that consist entirely of negative samples.")
        print(f"--before deletion: {self.pairs.shape}")
        label: pd.DataFrame = self.label[[self.col1["user"], self.col2["user"]]]
        label = label.rename(columns={self.col2["user"]: "positives"})
        self.pairs: pd.DataFrame
        self.pairs = self.pairs.merge(label, how="left")
        self.pairs["positives"] = self.pairs.apply(
            lambda row: 1 if row["positives"] in row[self.col2["user"]] else 0, axis=1)
        self.pairs = self.pairs[self.pairs["positives"] == 1]
        self.pairs.drop(columns=["positives"], inplace=True)
        print(f"--after deletion: {self.pairs.shape}")
