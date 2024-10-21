import logging

import numpy as np
import pandas as pd
import faiss


def evaluator(vector1, vector2, k=10):
    vector1 = np.array(vector1, dtype=np.float32)
    vector2 = np.array(vector2, dtype=np.float32)
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

    acc = {}
    for i in range(k):
        num = len(result[result['rank'] <= i])
        score = round(num / total, 6)
        logging.info(f"Acc@{i + 1}={num}/{total}={score}")
        print(f"Acc@{i + 1}={num}/{total}={score}")
        acc[f"Acc@{i + 1}"] = score

    distances = pd.DataFrame(data=distances, columns=[f"dis{i}" for i in range(1, k + 1)])
    indices = pd.DataFrame(data=indices, columns=[f"rank{i}" for i in range(1, k+1)])
    return distances, indices, acc
