import pandas as pd
from datetime import datetime

from libtrajectory.dataset.dataset_load import DataLoader
from libtrajectory.preprocessing.ST_trajectory_correlation.data_augmentation import DataAugmentation
from libtrajectory.preprocessing.ST_trajectory_correlation.determining_candidate_pairs import DeterminingCandidatePairs
from libtrajectory.preprocessing.ST_trajectory_correlation.feature_extraction import FeatureExtraction
from libtrajectory.preprocessing.abstract_preprocessor import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    def __init__(self, config, info):
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
        if self.info not in ["train", "test"]:
            raise print('Preprocessor param info not is ["train", "test"]')

        if self.info == "train":
            self.config["start_time"] = self.config["train"]["start_time"]
            self.config["end_time"] = self.config["train"]["end_time"]
        else:  # test
            self.config["start_time"] = self.config["test"]["start_time"]
            self.config["end_time"] = self.config["test"]["end_time"]

    def data_loading(self):
        print("data loading")
        start = datetime.now()
        data_loader = DataLoader()

        self.data1 = data_loader.reading(
            start_time=self.config["start_time"], end_time=self.config["end_time"],
            **self.config['data1'])
        self.col1 = self.config['data1']['columns']

        self.data2 = data_loader.reading(
            start_time=self.config["start_time"], end_time=self.config["end_time"],
            **self.config['data2'])
        self.col2 = self.config['data2']['columns']

        self.label = data_loader.reading_label(**self.config['label'])
        self.label["label"] = 1
        self.col_label = [self.col1["user"], self.col2["user"], "label"]
        print(f"label number : {self.label.shape}")
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def data_cleaning(self):
        print("data cleaning")
        start = datetime.now()

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
        if self.col1.get("coverage", None):
            self.data1[self.col1['coverage']] = self.data1[self.col1['coverage']].astype('category')
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
        if self.col2.get("coverage", None):
            self.data2[self.col2['coverage']] = self.data2[self.col2['coverage']].astype('category')

        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def data_augmentation(self):
        print("data augmentation")
        start = datetime.now()
        augmenter = DataAugmentation(
            data=self.data1[[self.col1['user'], self.col1['time']]],
            col={"user": self.col1['user'], "time": self.col1['time']})
        self.segment, self.col_segment = augmenter.max_length()
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

    def feature_engineering(self):
        print("feature engineering")
        start = datetime.now()

        print("--determining candidate pairs")
        config = self.config['pairs']
        pairs_class = DeterminingCandidatePairs(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2,
            front_time=config['front_time'], back_time=config['back_time'])

        self.pairs, self.col_pairs = pairs_class.sti_place_num(
            segment=self.segment, col_segment=self.col_segment,
            place_top=config['place_top'], num_top=config['num_top']
        )
        self.pairs.to_csv(f"./{self.info}_pairs.csv", index=False)
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

        if self.info == "train":
            self._drop_ns_group()

        print("--feature extraction")
        extractor = FeatureExtraction(
            data1=self.data1, col1=self.col1, data2=self.data2, col2=self.col2, **self.config['feature'])
        self.feature, self.col_feature = extractor.sequence(
            pairs=self.pairs, col_pairs=self.col_pairs)

        self.feature.to_csv(f"./{self.info}_feature.csv", index=False)
        print(f'Running time: {datetime.now() - start} Seconds', '\n')

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

    def label_sample(self):
        self.label = self.label.sample(1000, random_state=34078)

    # def test(self):
    #     feature_col = ["length1", "device1_num", "time1_diff", "count_distance1", "max_distance1", "length2",
    #                    "device2_num", "count_distance2", "max_distance2", "sti_length_50", "sti_user1_num_50",
    #                    "sti_user2_num_50", "sti_device1_num_50", "sti_device2_num_50", "sti_time1_diff_50",
    #                    "sti_count_distance1_50", "sti_max_distance1_50"]
    #     data_loader = DataLoader()
    #     self.label = data_loader.reading_label(**self.config['label'])
    #     self.label["label"] = 1
    #     self.col_label = ["face_user", "imsi_user", "label"]
    #
    #     train_feature = pd.read_csv("./train_feature.csv")
    #     train = train_feature.merge(self.label, how="left")
    #     train["label"].fillna(0, inplace=True)
    #     y_train = train["label"].to_list()
    #     X_train = train[feature_col]
    #
    #     test_feature = pd.read_csv("./test_feature.csv")
    #     test = test_feature.merge(self.label, how="left")
    #     test["label"].fillna(0, inplace=True)
    #     y_test = test["label"].to_list()
    #     X_test = test[feature_col]
    #     index = test[["face_user", "imsi_user", "segment"]]
    #     index.reset_index(drop=True, inplace=True)
    #
    #     return X_train, y_train, X_test, y_test, index
