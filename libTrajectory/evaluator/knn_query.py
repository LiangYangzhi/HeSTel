import logging

import numpy as np
import pandas as pd
from pyemd import emd
from sklearn.neighbors import KNeighborsRegressor


def dot_distance(vector1, vector2):
    return 1 / (1 + np.dot(vector1, vector2))


# year_month
year_month_matrix = np.array([[(abs(i - j) * 1) / 6 if abs(i - j) * 1 <= 6 else (12 - abs(i - j) * 1) / 6
                               for j in range(12)] for i in range(12)], dtype=np.float64)


def emd_year_month_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=year_month_matrix)
    return distance


# week_day
week_day_matrix = np.array([[(abs(i - j) * 1) / 3 if abs(i - j) * 1 <= 3 else (7 - abs(i - j) * 1) / 3
                         for j in range(7)] for i in range(7)], dtype=np.float64)


def emd_week_day_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=week_day_matrix)
    return distance


# month_day
month_day_matrix = np.array([[(abs(i - j) * 1) / 15 if abs(i - j) * 1 <= 15 else (31 - abs(i - j) * 1) / 15
                         for j in range(31)] for i in range(31)], dtype=np.float64)


def emd_month_day_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=month_day_matrix)
    return distance


# day_hour
day_hour_matrix = np.array([[(abs(i - j) * 1) / 12 if abs(i - j) * 1 <= 12 else (24 - abs(i - j) * 1) / 12
                         for j in range(24)] for i in range(24)], dtype=np.float64)


def emd_day_hour_distance(vector1, vector2):
    distance = 1 - emd(first_histogram=vector1, second_histogram=vector2, distance_matrix=day_hour_matrix)
    return distance


def evaluator(vector1, vector2, k=10, method='dot'):
    metric = {
        "dot": dot_distance,
        "year_month": emd_year_month_distance,
        "month_day": emd_month_day_distance,
        "week_day": emd_week_day_distance,
        "day_hour": emd_day_hour_distance
    }
    logging.info(f"k = {k}")
    knn_model = KNeighborsRegressor(n_neighbors=k, metric=metric[method])
    user_labels = np.arange(len(vector1))
    knn_model.fit(vector1, user_labels)
    distances, indices = knn_model.kneighbors(vector2, return_distance=True)

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