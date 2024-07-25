import logging
import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    logging.basicConfig(filename=f'./analysis.logs', format='%(asctime)s - %(message)s', level=logging.INFO)
    files = os.listdir('./original')
    files = pd.DataFrame(data=files, columns=['f'])
    colunms = ['uid', 'time', 'lon', 'lat']
    files['tra'] = files['f'].map(lambda f: pd.read_csv(f'./original/{f}', sep=',', header=None, names=colunms))
    files['tra'] = files['tra'].map(lambda d: d[['uid']])
    df = pd.concat(files.tra.to_list())
    df.reset_index(drop=True, inplace=True)

    logging.info(f"original vehicle num: {len(df.uid.unique())}")
    logging.info(f"original trajectory points num: {df.shape[0]}")

    columns = ['tid', 'time', 'lat', 'lon', 'stid']
    data1 = np.load(f"./data1.npy", allow_pickle=True)
    data1 = pd.DataFrame(data1, columns=columns).infer_objects()
    data2 = np.load(f"./data2.npy", allow_pickle=True)
    data2 = pd.DataFrame(data2, columns=columns).infer_objects()
    logging.info(f"data1 vehicle num: {len(data1.tid.unique())}")
    logging.info(f"data1 trajectory points num: {data1.shape[0]}")
    logging.info(f"data2 vehicle num: {len(data2.tid.unique())}")
    logging.info(f"data2 trajectory points num: {data2.shape[0]}")

    # dic1 = dict(data1.tid.value_counts())
    # arr1 = np.array(list(dic1.values()))
    # print("active")
    # print(f"mean: {np.mean(arr1)}")  # 113.23843585367663
    # print(f"median: {np.median(arr1)}")  # 49.0
    # print()
    #
    # dic2 = dict(data2.tid.value_counts())
    # arr2 = np.array(list(dic2.values()))
    # print("passive")
    # print(f"mean: {np.mean(arr2)}")  # 56.81961860948282
    # print(f"median: {np.median(arr2)}")  # 24.0
