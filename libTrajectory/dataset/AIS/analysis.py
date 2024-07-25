import logging
import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    logging.basicConfig(filename=f'./analysis.logs', format='%(asctime)s - %(message)s', level=logging.INFO)
    # data = []
    # for i in range(1, 10):
    #     files = os.listdir(f'./{i}')
    #     files = [f for f in files if "AIS" not in f]
    #     files = pd.DataFrame(data=files, columns=['f'])
    #     files['tra'] = files['f'].map(lambda f: pd.read_csv(f'./{i}/{f}', dtype={
    #         'uid': str, 'time': int, 'lat': float, 'lon': float, 'did': str}))
    #     df = pd.concat(files.tra.to_list())
    #     df.reset_index(drop=True, inplace=True)
    #     df = df[['uid', 'time']]
    #     data.append(df)
    # df = pd.concat(data)
    # logging.info(f"original vessel num: {len(df.uid.unique())}")
    # logging.info(f"original trajectory points num: {df.shape[0]}")

    columns = ['tid', 'time', 'lat', 'lon', 'stid']
    data1 = np.load(f"./data1.npy", allow_pickle=True)
    data1 = pd.DataFrame(data1, columns=columns).infer_objects()
    data2 = np.load(f"./data2.npy", allow_pickle=True)
    data2 = pd.DataFrame(data2, columns=columns).infer_objects()
    logging.info(f"data1 vessel num: {len(data1.tid.unique())}")
    logging.info(f"data1 trajectory points num: {data1.shape[0]}")
    logging.info(f"data2 vessel num: {len(data2.tid.unique())}")
    logging.info(f"data2 trajectory points num: {data2.shape[0]}")

    # dic1 = dict(data1.tid.value_counts())
    # arr1 = np.array(list(dic1.values()))
    # print("active")
    # print(f"mean: {np.mean(arr1)}")  # 661.7259171597633
    # print(f"median: {np.median(arr1)}")  # 129.0
    # print()
    #
    # dic2 = dict(data2.tid.value_counts())
    # arr2 = np.array(list(dic2.values()))
    # print("passive")
    # print(f"mean: {np.mean(arr2)}")  # 296.30141025641024
    # print(f"median: {np.median(arr2)}")  # 55.0
