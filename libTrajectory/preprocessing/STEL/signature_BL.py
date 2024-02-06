"""
<< Trajectory-Based Spatiotemporal Entity Linking >> 实验复现

signature
sequential signature：时空点作为词，grams设置为2，进行TF-IDF提取向量并进行L2 normalization
temporal signature：一天中的1h作为时间间隔，统计在每个时间间隔内出现的频率并进行L1 normalization
spatial signature：空间点作为词，进行TF-IDF提取向量进行L2 normalization
spatiotemporal signature：时间间隔+空间点作为词，进行TF-IDF提取向量进行L2 normalization。

similarity
sequential similarity：dot product
temporal similarity：(1- EMD) distance
spatial similarity: dot product
spatiotemporal similarity: dot product

base knn query
"""
import logging
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from pyemd import emd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk import ngrams
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor as Pre


class Preprocessor(Pre):
    def __init__(self, data_path, test_path={}):
        super(Preprocessor, self).__init__(data_path, test_path)
        self.loader()
        self.cleaner()

    def sequential(self):
        logging.info("sequential signature...")
        self.data1['seq'] = self.data1.apply(lambda row: (row.time, row.lat, row.lon), axis=1)
        group1 = self.data1.groupby("tid", as_index=False).agg({"seq": list})
        group1['seq'] = group1['seq'].map(lambda lis: list(ngrams(lis, 2)))
        group1['seq'] = group1['seq'].map(lambda s: str(s))

        self.data2['seq'] = self.data2.apply(lambda row: (row.time, row.lat, row.lon), axis=1)
        group2 = self.data2.groupby("tid", as_index=False).agg({"seq": list})
        group2['seq'] = group2['seq'].map(lambda lis: list(ngrams(lis, 2)))
        group2['seq'] = group2['seq'].map(lambda s: str(s))

        group = pd.concat([group1, group2])
        group.reset_index(drop=True, inplace=True)

        logging.info("sequential fit transform...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group['seq'])
        logging.info(f"data sequential matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        reduced_matrix = svd.fit_transform(matrix)
        logging.info(f"perform SVD dimensionality reduction after data shape: {reduced_matrix.shape}")

        group1['seq'] = group1.index.map(lambda i: normalize([reduced_matrix[i]], 'l2'))
        vector1 = group1[['userid', 'sequential']]
        group2['seq'] = group2.index.map(lambda i: normalize([reduced_matrix[i]], 'l2'))
        vector2 = group2[['userid', 'sequential']]

        logging.info("sequential signature competed")
        return vector1, vector2

    def temporal(self, inter=60 * 60):
        logging.info(f"temporal signature, time interval={inter}...")

        def time_interval(lis):
            vector = [0 for _ in range(24)]
            for i in lis:
                vector[i] += 1
            vector = np.array(vector, dtype=np.float64)
            vector_l1 = normalize([vector], 'l1')
            return vector_l1

        self.data1['tem'] = self.data1['time'].parallel_map(lambda t: t // inter)
        self.data1['tem'] = self.data1['tem'].astype(int)
        group1 = self.data1.groupby("tid", as_index=False).agg({"tem": list})
        group1['teml'] = group1['tem'].map(time_interval)
        vector1 = group1[['userid', 'tem']]

        self.data2['tem'] = self.data2['time'].parallel_map(lambda t: t // inter)
        self.data2['tem'] = self.data2['tem'].astype(int)
        group2 = self.data2.groupby("tid", as_index=False).agg({"tem": list})
        group2['teml'] = group2['tem'].map(time_interval)
        vector2 = group2[['userid', 'tem']]

        logging.info(f"temporal signature completed")
        return vector1, vector2

    def spatial(self):
        logging.info(f"spatial signature...")
