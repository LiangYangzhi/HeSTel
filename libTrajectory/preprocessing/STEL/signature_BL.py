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
        svd = TruncatedSVD(n_components=64, algorithm='arpack')
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
            np.save(f'{log_path}sequential_{self.test_path[k].split("/")[-1]}_embedding1.npy', embedding1)
            np.save(f'{log_path}sequential_{self.test_path[k].split("/")[-1]}_embedding2.npy', embedding2)
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_tem(self, data: pd.DataFrame):
        def time_interval(lis):
            v = [0 for _ in range(24)]
            for i in lis:
                t_tuple = datetime.fromtimestamp(i)
                hour = t_tuple.hour
                v[hour] += 1
            v = np.array(v, dtype=np.float64)
            v_l1 = normalize([v], 'l1')
            return v_l1

        group = data.groupby("tid", as_index=False).agg({"time": list})
        group['tem'] = group['time'].map(time_interval)  # parallel_map
        return group

    def temporal(self):
        logging.info(f"temporal signature, time interval={self.inter}...")
        # t0 = min([self.data1.time.min(), self.data2.time.min()])
        # self.t0 = datetime.fromtimestamp(t0).day
        # t1 = max([self.data1.time.max(), self.data2.time.max()])
        # self.t1 = datetime.fromtimestamp(t1).day
        test_data = {}
        for k, tid in self.test_data.items():
            logging.info(f"{k}, data1--->temporal vector....")
            test1 = self.data1[self.data1['tid'].isin(tid)].copy()
            test1 = self._deal_tem(test1)
            logging.info(f"{k}, data2--->temporal vector....")
            test2 = self.data2[self.data2['tid'].isin(tid)].copy()
            test2 = self._deal_tem(test2)

            embedding1, embedding2 = self._vector_format(test1, test2, name='tem')
            np.save(f'{log_path}temporal_{self.test_path[k].split("/")[-1]}_embedding1.npy', embedding1)
            np.save(f'{log_path}temporal_{self.test_path[k].split("/")[-1]}_embedding2.npy', embedding2)
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_spa(self, data: pd.DataFrame):
        data['point'] = data.parallel_apply(lambda row: (row.lat, row.lon), axis=1)
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
            np.save(f'{log_path}spatial_{self.test_path[k].split("/")[-1]}_embedding1.npy', embedding1)
            np.save(f'{log_path}spatial_{self.test_path[k].split("/")[-1]}_embedding2.npy', embedding2)
            test_data[k] = [embedding1, embedding2]
        return test_data

    def _deal_st(self, data: pd.DataFrame):
        data['tem'] = data['time'].map(lambda t: t // self.inter)
        data['st'] = data.apply(lambda row: (row.tem, row.lat, row.lon), axis=1)
        group = data.groupby("tid", as_index=False).agg({"st": list})
        group['st'] = group['st'].map(lambda s: str(s))
        group.reset_index(drop=True, inplace=True)
        return group

    def spatiotemporal(self):
        logging.info(f"spatiotemporal signature...")
        logging.info("data1 spatiotemporal...")
        group1 = self._deal_st(self.data1)
        group1['data'] = 'data1'
        logging.info("data2 spatiotemporal...")
        group2 = self._deal_st(self.data2)
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
            np.save(f'{log_path}spatiotemporal_{self.test_path[k].split("/")[-1]}_embedding1.npy', embedding1)
            np.save(f'{log_path}spatiotemporal_{self.test_path[k].split("/")[-1]}_embedding2.npy', embedding2)
            test_data[k] = [embedding1, embedding2]
        return test_data
