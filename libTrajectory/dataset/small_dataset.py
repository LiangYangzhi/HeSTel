import pickle
import random
import shutil
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def small():
    # with open(f"{path}train_tid.pkl", "rb") as f:
    #     train_tid = pickle.load(f)
    # train_tid = [f"{tid}" for tid in train_tid]
    # small_train_tid = random.sample(train_tid, 7000)
    # with open(f"{path}small_train_tid.pkl", 'wb') as f:
    #     pickle.dump(small_train_tid, f)

    small_path = path.replace('./', './small_')
    # shutil.copyfile(f"{path}small_train_tid.pkl", f"{small_path}train_tid.pkl")
    # shutil.copyfile(f"{path}test1K.csv", f"{small_path}test1K.csv")
    # shutil.copyfile(f"{path}test3K.csv", f"{small_path}test3K.csv")
    # shutil.copyfile(f"{path}enhance_tid.csv", f"{small_path}enhance_tid.csv")

    with open(f"{small_path}train_tid.pkl", "rb") as f:
        train_tid = pickle.load(f)
    train_tid = [f"{tid}" for tid in train_tid]
    test1 = pd.read_csv(f"{small_path}test1K.csv", usecols=['tid'], dtype={'tid': str})
    test1 = test1.tid.unique().tolist()
    test3 = pd.read_csv(f"{small_path}test3K.csv", usecols=['tid'], dtype={'tid': str})
    test3 = test3.tid.unique().tolist()
    tids = train_tid + test1 + test3

    # columns = ['tid', 'time', 'lat', 'lon', 'did']
    # arr1 = np.load(f"{path}data1.npy", allow_pickle=True)
    # data1 = pd.DataFrame(arr1, columns=columns).infer_objects()
    # data1 = data1[data1['tid'].isin(tids)]
    # arr1 = data1.to_numpy()
    # np.save(f"{small_path}data1.npy", arr1)
    #
    # arr2 = np.load(f"{path}data2.npy", allow_pickle=True)
    # data2 = pd.DataFrame(arr2, columns=columns).infer_objects()
    # data2 = data2[data2['tid'].isin(tids)]
    # arr2 = data2.to_numpy()
    # np.save(f"{small_path}data2.npy", arr2)

    # print("graph...")
    # os.mkdir(f"{small_path}graph1")
    # os.mkdir(f"{small_path}graph2")
    # for tid in tqdm(tids):
    #     shutil.copyfile(f"{path}graph1/{tid}.npz", f"{small_path}graph1/{tid}.npz")
    #     shutil.copyfile(f"{path}graph2/{tid}.npz", f"{small_path}graph2/{tid}.npz")

    enhance_tid = pd.read_csv(f"{small_path}enhance_tid.csv")
    enhance_tid = enhance_tid[enhance_tid['tid'].isin(tids)]
    for name, folder in zip(['ps1', 'ps2', 'ns1', 'ns2'], ["ps_graph1", "ps_graph2", "ns_graph1", "ns_graph2"]):
        print(f"{name}...")
        os.mkdir(f"{small_path}{folder}")
        enhance_tid[name] = enhance_tid[name].map(lambda x: eval(x))
        ns1_tids = enhance_tid[name].values
        for ns1_tid in tqdm(ns1_tids):
            for tid in ns1_tid:
                shutil.copyfile(f"{path}{folder}/{tid}.npz", f"{small_path}{folder}/{tid}.npz")


if __name__ == "__main__":
    path = "./taxi/"
    small()
