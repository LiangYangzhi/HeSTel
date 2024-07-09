import math
import os
import random

import pandas as pd


if __name__ == "__main__":
    from datetime import datetime
    import time

    # 获取当前时间的时间戳
    timestamp = 1704071650

    # 将时间戳转换为datetime对象
    dt_object = datetime.fromtimestamp(timestamp)

    # 获取当前日期是一年中的第几周
    week_number = dt_object.isocalendar()[1]

    print(week_number)

    # ns_tid
    # df = pd.read_csv("./libTrajectory/dataset/ais/enhance_tid.csv")
    # df['ns1'] = df['ns1'].map(lambda x: eval(x))
    # df['ns2'] = df['ns2'].map(lambda x: eval(x))

    # def shuffle(lis):
    #     lis = eval(lis)
    #     random.shuffle(lis)
    #     return lis[:8]
    # df['ps1'] = df['ps1'].map(shuffle)
    # df['ps2'] = df['ps2'].map(shuffle)
    # df.to_csv(f"./libTrajectory/dataset/ais/random_enhance_tid.csv", index=False)

    # df = pd.read_csv("./libTrajectory/dataset/ais/enhance_tid.csv")
    # df['ps1'] = df['ps1'].map(lambda x: eval(x))
    # ps1 = [lis for lis in df.ps1.values if lis]
    # files = os.listdir("./libTrajectory/dataset/ais/new_ps_graph1/")
    # files = {i: 1for i in files}
    # for lis in ps1:
    #     for ps1_tid in lis:
    #         if f"{ps1_tid}.npz" in files:
    #             pass
    #         else:
    #             print(f"{ps1_tid}.npz")
    #
    # df['ps2'] = df['ps2'].map(lambda x: eval(x))
    # ps2 = [lis for lis in df.ps2.values if lis]
    # files = os.listdir("./libTrajectory/dataset/ais/new_ps_graph2/")
    # files = {i: 1for i in files}
    # for lis in ps2:
    #     for ps2_tid in lis:
    #         if f"{ps2_tid}.npz" in files:
    #             pass
    #         else:
    #             print(f"{ps2_tid}.npz")

    # df['ns1'] = df['ns1'].map(lambda x: eval(x))
    # ns1 = [lis for lis in df.ns1.values if lis]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_graph1/")
    # files = {i: 1for i in files}
    # for lis in ns1:
    #     for ns1_tid in lis:
    #         if f"{ns1_tid}.npz" in files:
    #             pass
    #         else:
    #             print(f"{ns1_tid}.npz")

    # df['ns2'] = df['ns2'].map(lambda x: eval(x))
    # ns2 = [lis for lis in df.ns2.values if lis]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_graph2/")
    # files = {i: 1for i in files}
    # for lis in ns2:
    #     for ns2_tid in lis:
    #         if f"{ns2_tid}.npz" in files:
    #             pass
    #         else:
    #             print(f"{ns2_tid}.npz")

    # file_path = f"{self.path}/ns_traj{data}/{ns}.csv"
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # file_path = f"{self.path}/ns_graph{data}/{ns}.npz"
    # if os.path.exists(file_path):
    #     os.remove(file_path)

