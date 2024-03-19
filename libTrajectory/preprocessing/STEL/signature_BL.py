"""
<< Trajectory-Based Spatiotemporal Entity Linking >> 实验复现

signature
sequential signature：时空点作为词，grams设置为2，进行TF-IDF提取向量并进行L2 normalization
temporal signature：一天中的1h作为时间间隔，统计在每个时间间隔内出现的频率并进行L1 normalization
spatial signature：空间点作为词，进行TF-IDF提取向量进行L2 normalizations
spatiotemporal signature：时间间隔+空间点作为词，进行TF-IDF提取向量进行L2 normalization。

similarity
sequential similarity：dot product
temporal similarity：(1- EMD) distance
spatial similarity: dot product
spatiotemporal similarity: dot product

base knn query
"""
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk import ngrams
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor as Pre

log_path = "./libTrajectory/logs/STEL_BL/"


class Preprocessor(Pre):
    def __init__(self, data_path, test_path={}):
        self.inter = 60 * 60
        super(Preprocessor, self).__init__(data_path, test_path)
        self.loader()
        self.cleaner()

        self.test_data = {}
        for k, v in self.test_path.items():
            tid = pd.read_csv(f"{v}", dtype={'tid': str, 'time': int}).tid.unique().tolist()
            self.test_data[k] = tid

    def _vector_format(self, v1, v2, name):
        tid = v1.tid.unique().tolist()
        e1 = []
        e2 = []
        for i in tid:
            e1.append(v1[v1['tid'] == i][name].values[0][0])
            e2.append(v2[v2['tid'] == i][name].values[0][0])
        e1 = np.array(e1)
        e2 = np.array(e2)
        return e1, e2

    def _deal_seq(self, data: pd.DataFrame):
        data['seq'] = data.apply(lambda row: (row.time, row.lat, row.lon), axis=1)
        group = data.groupby("tid", as_index=False).agg({"seq": list})
        group['seq'] = group['seq'].map(lambda lis: list(ngrams(lis, 2)))
        group['seq'] = group['seq'].map(lambda s: str(s))
        return group

    def sequential(self):
        logging.info("sequential signature...")
        logging.info("data1 sequential...")
        group1 = self._deal_seq(self.data1)
        group1['data'] = 'data1'
        logging.info("data2 sequential...")
        group2 = self._deal_seq(self.data2)
        group2['data'] = 'data2'
        group = pd.concat([group1, group2])
        group.reset_index(drop=True, inplace=True)
        group1 = group[group['data'] == 'data1']
        group1 = group1.drop(columns=['data'])
        group2 = group[group['data'] == 'data2']
        group2 = group2.drop(columns=['data'])

        logging.info("sequential fit transform...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group['seq'])
        logging.info(f"data sequential matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=128, algorithm='arpack')
        matrix = svd.fit_transform(matrix)
        logging.info(f"perform SVD dimensionality reduction after data shape: {matrix.shape}")

        test_data = {}
        for k, tid in self.test_data.items():
            logging.info(f"{k}, data1--->sequential vector....")
            test1 = group1[group1['tid'].isin(tid)].copy()
            test1['seq'] = test1.index.map(lambda i: normalize([matrix[i]], 'l2'))   # [matrix[i]]  matrix.getrow(i).toarray()

            logging.info(f"{k}, data2--->sequential vector....")
            test2 = group2[group2['tid'].isin(tid)].copy()
            test2['seq'] = test2.index.map(lambda i: normalize([matrix[i]], 'l2'))

            embedding1, embedding2 = self._vector_format(test1, test2, name='seq')
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_tem(self, data: pd.DataFrame, method):  # year_month month_day week_day day_hour

        def year_month_interval(lis):
            v = [0 for _ in range(12)]  # tm_mon: 12, tm_wday: 7, tm_hour: 24
            for t in lis:
                ind = time.localtime(t).tm_mon - 1  # tm_mon - 1  , tm_wday,  tm_hour
                v[ind] += 1
            v = np.array(v, dtype=np.float64)
            v_l1 = normalize([v], 'l1')
            return v_l1

        def month_day_interval(lis):
            v = [0 for _ in range(31)]
            for t in lis:
                ind = time.localtime(t).tm_mday - 1
                v[ind] += 1
            v = np.array(v, dtype=np.float64)
            v_l1 = normalize([v], 'l1')
            return v_l1

        def week_day_interval(lis):
            v = [0 for _ in range(7)]
            for t in lis:
                ind = time.localtime(t).tm_wday
                v[ind] += 1
            v = np.array(v, dtype=np.float64)
            v_l1 = normalize([v], 'l1')
            return v_l1

        def day_hour_interval(lis):
            v = [0 for _ in range(24)]
            for t in lis:
                ind = time.localtime(t).tm_hour
                v[ind] += 1
            v = np.array(v, dtype=np.float64)
            v_l1 = normalize([v], 'l1')
            return v_l1

        group = data.groupby("tid", as_index=False).agg({"time": list})
        if method == "year_month":
            group['tem'] = group['time'].map(year_month_interval)  # parallel_map
        elif method == "month_day":
            group['tem'] = group['time'].map(month_day_interval)
        elif method == "week_day":
            group['tem'] = group['time'].map(week_day_interval)
        elif method == "day_hour":
            group['tem'] = group['time'].map(day_hour_interval)
        return group

    def temporal(self, method="month_day"):  # year_month month_day week_day day_hour
        logging.info(f"temporal signature, method={method}...")
        test_data = {}
        for k, tid in self.test_data.items():
            logging.info(f"{k}, data1--->temporal vector....")
            test1 = self.data1[self.data1['tid'].isin(tid)].copy()
            test1 = self._deal_tem(test1, method=method)
            logging.info(f"{k}, data2--->temporal vector....")
            test2 = self.data2[self.data2['tid'].isin(tid)].copy()
            test2 = self._deal_tem(test2, method=method)

            embedding1, embedding2 = self._vector_format(test1, test2, name='tem')
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_spa(self, data: pd.DataFrame):
        data['point'] = data.apply(lambda row: (row.lat, row.lon), axis=1)   # parallel_apply
        group = data.groupby("tid", as_index=False).agg({"point": list})
        group['point'] = group['point'].map(lambda s: str(s))
        group.reset_index(drop=True, inplace=True)
        return group

    def spatial(self):
        logging.info(f"spatial signature...")
        logging.info("data1 spatial...")
        group1 = self._deal_spa(self.data1)
        group1['data'] = 'data1'
        logging.info("data2 spatial...")
        group2 = self._deal_spa(self.data2)
        group2['data'] = 'data2'
        group = pd.concat([group1, group2])
        group.reset_index(drop=True, inplace=True)

        group1 = group[group['data'] == 'data1']
        group1 = group1.drop(columns=['data'])
        group2 = group[group['data'] == 'data2']
        group2 = group2.drop(columns=['data'])

        logging.info("spatial fit transform...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group['point'])
        logging.info(f"data spatial matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=128, algorithm='arpack')
        matrix = svd.fit_transform(matrix)
        logging.info(f"perform SVD dimensionality reduction after data shape: {matrix.shape}")

        test_data = {}
        for k, tid in self.test_data.items():
            logging.info(f"{k}, data1--->spatial vector....")
            test1 = group1[group1['tid'].isin(tid)].copy()
            test1['spatial'] = test1.index.map(lambda i: normalize([matrix[i]], 'l2'))

            logging.info(f"{k}, data2--->spatial vector....")
            test2 = group2[group2['tid'].isin(tid)].copy()
            test2['spatial'] = test2.index.map(lambda i: normalize([matrix[i]], 'l2'))

            embedding1, embedding2 = self._vector_format(test1, test2, name='spatial')
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_st(self, data: pd.DataFrame, method):  # year_month month_day week_day day_hour
        if method == "year_month":
            data['tem'] = data['time'].map(lambda t: time.localtime(t).tm_mon)
        elif method == "month_day":
            data['tem'] = data['time'].map(lambda t: time.localtime(t).tm_mday)
        elif method == "week_day":
            data['tem'] = data['time'].map(lambda t: time.localtime(t).tm_wday)
        elif method == "day_hour":
            data['tem'] = data['time'].map(lambda t: time.localtime(t).tm_hour)

        data['st'] = data.apply(lambda row: (row.tem, row.lat, row.lon), axis=1)
        group = data.groupby("tid", as_index=False).agg({"st": list})
        group['st'] = group['st'].map(lambda s: str(s))
        group.reset_index(drop=True, inplace=True)
        return group

    def spatiotemporal(self, method="month_day"):  # year_month month_day week_day day_hour
        logging.info(f"spatiotemporal signature, method={method}...")
        logging.info("data1 spatiotemporal...")
        group1 = self._deal_st(self.data1, method)
        group1['data'] = 'data1'
        logging.info("data2 spatiotemporal...")
        group2 = self._deal_st(self.data2, method)
        group2['data'] = 'data2'
        group = pd.concat([group1, group2])
        group.reset_index(drop=True, inplace=True)

        group1 = group[group['data'] == 'data1']
        group1 = group1.drop(columns=['data'])
        group2 = group[group['data'] == 'data2']
        group2 = group2.drop(columns=['data'])

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group['st'])
        logging.info(f"data spatiotemporal matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=128, algorithm='arpack')
        matrix = svd.fit_transform(matrix)
        logging.info(f"perform SVD dimensionality reduction after data shape: {matrix.shape}")

        test_data = {}
        for k, tid in self.test_data.items():
            logging.info(f"{k}, data1--->spatiotemporal vector....")
            test1 = group1[group1['tid'].isin(tid)].copy()
            test1['st'] = test1.index.map(lambda i: normalize([matrix[i]], 'l2'))

            logging.info(f"{k}, data2--->spatiotemporal vector....")
            test2 = group2[group2['tid'].isin(tid)].copy()
            test2['st'] = test2.index.map(lambda i: normalize([matrix[i]], 'l2'))

            embedding1, embedding2 = self._vector_format(test1, test2, name='st')
            test_data[k] = [embedding1, embedding2]
        return test_data
