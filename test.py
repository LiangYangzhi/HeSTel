import math
import os
import pandas as pd


# if __name__ == "__main__":
    # df = pd.read_csv("./libTrajectory/dataset/ais/enhance_ns.csv")
    # df['ns1'] = df['ns1'].map(lambda x: eval(x))
    #
    # ns1 = [lis for lis in df.ns1.values if lis]
    # ns1 = sum(ns1, [])
    #
    # ns_traj1 = [f"{i}.csv" for i in ns1]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_traj1/")
    # print(files[0], len(files), ns_traj1[0], len(ns_traj1))
    # print(f"files - ns_traj1 : {len(list(set(files) - set(ns_traj1)))}")
    # print(f"ns_traj1 - files : {len(list(set(ns_traj1) - set(files)))}")
    #
    # ns_graph1 = [f"{i}.npz" for i in ns1]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_graph1/")
    # print(files[0], len(files), ns_graph1[0], len(ns_graph1))
    # print(f"files - ns_graph1 : {len(list(set(files) - set(ns_graph1)))}")
    # print(f"ns_graph1 - files : {len(list(set(ns_graph1) - set(files)))}")
    #
    # df['ns2'] = df['ns2'].map(lambda x: eval(x))
    # ns2 = [lis for lis in df.ns2.values if lis]
    # ns2 = sum(ns2, [])
    #
    # ns_traj2 = [f"{i}.csv" for i in ns2]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_traj2/")
    # print(files[0], len(files), ns_traj2[0], len(ns_traj2))
    # print(f"files - ns_traj2 : {len(list(set(files) - set(ns_traj2)))}")
    # print(f"ns_traj2 - files : {len(list(set(ns_traj2) - set(files)))}")
    #
    # ns_graph2 = [f"{i}.npz" for i in ns2]
    # files = os.listdir("./libTrajectory/dataset/ais/ns_graph2/")
    # print(files[0], len(files), ns_graph2[0], len(ns_graph2))
    # print(f"files - ns_graph2 : {len(list(set(files) - set(ns_graph2)))}")
    # print(f"ns_graph2 - files : {len(list(set(ns_graph2) - set(files)))}")

    # file_path = f"{self.path}/ns_traj{data}/{ns}.csv"
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # file_path = f"{self.path}/ns_graph{data}/{ns}.npz"
    # if os.path.exists(file_path):
    #     os.remove(file_path)

