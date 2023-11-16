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
from pandarallel import pandarallel


def load_data():
    print("data loading...")
    data9001 = pd.read_csv("../dataset/MTAD/9001.csv")
    data9001['userid'] = data9001['userid'].map(lambda u: u + '_9001')
    data9001.sort_values("time", inplace=True)
    print(f"data9001 number: {len(data9001.userid.unique())}")
    print(f"data9001 shape: {data9001.shape}")

    data9002 = pd.read_csv("../dataset/MTAD/9002.csv")
    data9002['userid'] = data9002['userid'].map(lambda u: u + '_9002')
    data9002.sort_values("time", inplace=True)
    print(f"data9002 number: {len(data9002.userid.unique())}")
    print(f"data9002 shape: {data9002.shape}")

    data = pd.concat([data9001, data9002])
    return data


def data_split(data, name):
    feature9001 = data.query("userid.str.contains('_9001')", engine='python')
    feature9001.reset_index(drop=True, inplace=True)
    feature9002 = data.query("userid.str.contains('_9002')", engine='python')
    feature9002.reset_index(drop=True, inplace=True)

    user9001 = {i: u for i, u in enumerate(feature9001.userid.tolist())}
    user9002 = {i: u for i, u in enumerate(feature9002.userid.tolist())}

    vector9001 = np.concatenate(feature9001[name], axis=0)
    vector9002 = np.concatenate(feature9002[name], axis=0)

    print(f"user9001 number: {len(user9001)}")
    print(f"vector9001 shape: {vector9001.shape}")
    print(f"user9002 number: {len(user9002)}")
    print(f"vector9002 shape: {vector9002.shape}")
    return user9001, user9002, vector9001, vector9002


def dot_distance(vector1, vector2):
    return 1 / (1 + np.dot(vector1, vector2))


time_matrix = np.array([[(abs(i - j) * 1) / 12 if abs(i - j) * 1 <= 12 else (24 - abs(i - j) * 1) / 12
                         for j in range(24)] for i in range(24)], dtype=np.float64)


def emd_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=time_matrix)
    return distance


def knn_query(data, name):
    user9001, user9002, vector9001, vector9002 = data_split(data, name)
    distance = {"sequential": dot_distance, 
                "temporal": emd_distance,
                "spatial": dot_distance,
                "spatiotemporal": dot_distance}

    k = 5
    print(f"k = {k}")
    knn_model = KNeighborsRegressor(n_neighbors=k, metric=distance[name])
    user_labels = np.arange(len(user9001))
    knn_model.fit(vector9001, user_labels)

    result = {"userid9001": [], "userid9002": [], "distance": [], "rank": []}
    for i_9002, u_9002 in user9002.items():
        query_vector = np.array([vector9002[i_9002]])
        distances, indices = knn_model.kneighbors(query_vector, return_distance=True)
        for rank, i_9001 in enumerate(indices[0]):
            result['userid9001'].append(user9001[i_9001])
            result["userid9002"].append(u_9002)
            result['distance'].append(distances[0][rank])
            result['rank'].append(rank + 1)

        print("\r", end="")
        print(f"file number: {i_9002}/{len(user9002)}", end="")
        sys.stdout.flush()
    print()

    result = pd.DataFrame(result)
    result['label'] = result.apply(lambda row: 1 if row.userid9001[:-2] == row.userid9002[:-2] else 0, axis=1)
    return result


def evaluator(result):
    user_num = len(result.userid9002.unique())
    top1_num = result.query("(label == 1) & (rank <= 1)").shape[0]
    print(f"top1: {top1_num}/{user_num} = {top1_num / user_num}")

    top3_num = result.query("(label == 1) & (rank <= 3)").shape[0]
    print(f"top3: {top3_num}/{user_num} = {top3_num / user_num}")

    top5_num = result.query("(label == 1) & (rank <= 5)").shape[0]
    print(f"top5: {top5_num}/{user_num} = {top5_num / user_num}")


class Signature(object):
    def __init__(self, data):
        print("data structure signature feature...")
        self.data = data
        self.data.reset_index(drop=True, inplace=True)

    def sequential(self):
        pandarallel.initialize()
        self.data['sequential'] = self.data.parallel_apply(lambda row: (row.time, row.lat, row.lon), axis=1)
        group_user = self.data.groupby("userid", as_index=False).agg({"sequential": list})
        group_user['sequential'] = group_user['sequential'].map(lambda lis: list(ngrams(lis, 2)))
        group_user['sequential'] = group_user['sequential'].map(lambda s: str(s))
        group_user.reset_index(drop=True, inplace=True)

        print("sequential fit transform...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group_user['sequential'])
        print(f"data sequential matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        reduced_matrix = svd.fit_transform(matrix)
        print(f"perform SVD dimensionality reduction after data shape: {reduced_matrix.shape}")

        group_user['sequential'] = group_user.index.map(lambda i: normalize([reduced_matrix[i]], 'l2'))
        data = group_user[['userid', 'sequential']]
        return data

    def temporal(self):
        interval = 60 * 60  # 1h = 60 * 60 s
        print(f"time interval: {interval / 60 / 60}h")
        pandarallel.initialize()
        self.data['temporal'] = self.data['time'].parallel_map(lambda t: t // interval)
        self.data['temporal'] = self.data['temporal'].astype(int)
        group_user = self.data.groupby("userid", as_index=False).agg({"temporal": list})

        def time_interval(lis):
            vector = [0 for _ in range(24)]
            for i in lis:
                vector[i] += 1
            vector = np.array(vector, dtype=np.float64)
            vector_l1 = normalize([vector], 'l1')
            return vector_l1

        group_user['temporal'] = group_user['temporal'].map(time_interval)
        data = group_user[['userid', 'temporal']]
        return data

    def spatial(self):
        pandarallel.initialize()
        self.data['point'] = self.data.parallel_apply(lambda row: (row.lat, row.lon), axis=1)
        group_user = self.data.groupby("userid", as_index=False).agg({"point": list})
        group_user['point'] = group_user['point'].map(lambda s: str(s))
        group_user.reset_index(drop=True, inplace=True)

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group_user['point'])
        print(f"data spatial matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        reduced_matrix = svd.fit_transform(matrix)
        print(f"perform SVD dimensionality reduction after data shape: {reduced_matrix.shape}")

        group_user['spatial'] = group_user.index.map(lambda i: normalize([reduced_matrix[i]], 'l2'))
        data = group_user[['userid', 'spatial']]
        return data

    def spatiotemporal(self):
        pandarallel.initialize()
        self.data['spatiotemporal'] = self.data.parallel_apply(lambda row: (row.temporal, row.lat, row.lon), axis=1)
        group_user = self.data.groupby("userid", as_index=False).agg({"spatiotemporal": list})
        group_user['spatiotemporal'] = group_user['spatiotemporal'].map(lambda s: str(s))
        group_user.reset_index(drop=True, inplace=True)

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(group_user['spatiotemporal'])
        print(f"data spatiotemporal matrix shape {matrix.shape}")
        # perform SVD dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        reduced_matrix = svd.fit_transform(matrix)
        print(f"perform SVD dimensionality reduction after data shape: {reduced_matrix.shape}")
        
        group_user['spatiotemporal'] = group_user.index.map(lambda i: normalize([reduced_matrix[i]], 'l2'))
        data = group_user[['userid', 'spatiotemporal']]
        return data


def pipeline():
    dir_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logs_path = f"./logs/STEL_0/{dir_name}"
    os.makedirs(logs_path)

    t0 = datetime.now()
    signature = Signature(load_data())

    print("sequential...")
    feature = signature.sequential()
    result = knn_query(feature, "sequential")
    result.to_csv(f"{logs_path}/sequential_result.csv", index=False)
    evaluator(result)
    print(f'Running time: {datetime.now() - t0} Seconds', '\n')
    
    print("temporal...")
    feature = signature.temporal()
    result = knn_query(feature, "temporal")
    result.to_csv(f"{logs_path}/temporal_result.csv", index=False)
    evaluator(result)
    print(f'Running time: {datetime.now() - t0} Seconds', '\n')
    
    print("spatial...")
    feature = signature.spatial()
    result = knn_query(feature, "spatial")
    result.to_csv(f"{logs_path}/spatial_result.csv", index=False)
    evaluator(result)
    print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    print("spatiotemporal...")
    feature = signature.spatiotemporal()
    result = knn_query(feature, "spatiotemporal")
    result.to_csv(f"{logs_path}/spatiotemporal_result.csv", index=False)
    evaluator(result)
    print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    if os.path.exists('./STEL_1.logs'):
        shutil.copy('./STEL_1.logs', f"{logs_path}/STEL_1.logs")


if __name__ == "__main__":
    pipeline()
