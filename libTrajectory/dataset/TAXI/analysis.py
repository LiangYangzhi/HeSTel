import numpy as np
import pandas as pd


if __name__ == "__main__":
    columns = ['tid', 'time', 'lat', 'lon', 'stid']
    data1 = np.load(f"./data1.npy", allow_pickle=True)
    data1 = pd.DataFrame(data1, columns=columns).infer_objects()
    data2 = np.load(f"./data2.npy", allow_pickle=True)
    data2 = pd.DataFrame(data2, columns=columns).infer_objects()

    dic1 = dict(data1.tid.value_counts())
    arr1 = np.array(list(dic1.values()))
    print("active")
    print(f"mean: {np.mean(arr1)}")  # 113.23843585367663
    print(f"median: {np.median(arr1)}")  # 49.0
    print()

    dic2 = dict(data2.tid.value_counts())
    arr2 = np.array(list(dic2.values()))
    print("passive")
    print(f"mean: {np.mean(arr2)}")  # 56.81961860948282
    print(f"median: {np.median(arr2)}")  # 24.0
