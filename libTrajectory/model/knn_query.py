import logging

import numpy as np
import pandas as pd
from pyemd import emd
from sklearn.neighbors import KNeighborsRegressor


def dot_distance(vector1, vector2):
    return 1 / (1 + np.dot(vector1, vector2))


time_matrix = np.array([[(abs(i - j) * 1) / 12 if abs(i - j) * 1 <= 12 else (24 - abs(i - j) * 1) / 12
                         for j in range(24)] for i in range(24)], dtype=np.float64)


def emd_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=time_matrix)
    return distance


def knn_query(vector1, vector2, distance='dot', k=5):
    distance_def = {"dot": dot_distance, "emd": emd_distance}
    logging.info(f"k = {k}")
    knn_model = KNeighborsRegressor(n_neighbors=k, metric=distance_def[distance])
    user_labels = np.arange(len(vector1))
    knn_model.fit(vector1, user_labels)
    distances, indices = knn_model.kneighbors(vector2, return_distance=True)
    print(indices)

    result = []
    for i, arr in enumerate(indices):
        try:
            result.append(np.where(arr == i)[0][0])
        except IndexError:
            result.append(k+1)
    result = pd.DataFrame(data=result, columns=['rank'])
    total = len(result)
    for i in range(k):
        score = len(result[result['rank'] <= i])
        logging.info(f"top{i+1}={score}/{total}={round(score/total, 6)}")

    distances = pd.DataFrame(data=distances, columns=[f"dis{i}" for i in range(1, k + 1)])
    indices = pd.DataFrame(data=indices, columns=[f"rank{i}" for i in range(1, k+1)])
    return distances, indices