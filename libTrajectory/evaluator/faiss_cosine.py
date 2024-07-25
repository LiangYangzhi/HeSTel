import logging

import numpy as np
import pandas as pd
import faiss


def evaluator(vector1, vector2, k=10):
    logging.info(f"k = {k}")
    admin = vector1.shape[1]
    indexIP = faiss.IndexFlatIP(admin)  # L2 normalization after inner product == cosine
    faiss.Index
    indexIP.add(vector1)
    distances, indices = indexIP.search(vector2, k)

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
        logging.info(f"top{i+1} = {score} / {total} = {round(score/total, 6)}")
        print(f"top{i+1} = {score} / {total} = {round(score/total, 6)}")

    distances = pd.DataFrame(data=distances, columns=[f"dis{i}" for i in range(1, k + 1)])
    indices = pd.DataFrame(data=indices, columns=[f"rank{i}" for i in range(1, k+1)])
    return distances, indices
