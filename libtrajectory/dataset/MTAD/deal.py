import os
import re
import sys

import pandas as pd


def data_merging():
    # data1 = pd.read_csv("./数据集level1/量测场景/场景-0.csv")
    # data2 = pd.read_csv("./数据集level1/真实场景/场景-0.csv")
    # data3 = pd.read_csv("./数据集level1/关联表/关联结果-0.csv")
    #
    # data_merge2 = pd.merge(data1, data3, how='outer')
    # print(data_merge2.info())
    # data_merge2 = data_merge2.query("(time >= t_s) & (time <= t_e)")
    # print(data_merge2.info())

    # 获取当前目录下的所有文件和文件夹
    for i in [1, 2]:
        files = os.listdir(f"./数据集level{i}/关联表/")
        for file in files:
            path1 = f"./数据集level{i}/量测场景/{file.replace('关联结果', '场景')}"
            # path2 = f"./数据集level{i}/真实场景/{file.replace('关联结果', '场景')}"
            path3 = f"./数据集level{i}/关联表/{file}"
            print(f"{path1},     {path3}")
            data1 = pd.read_csv(path1)
            # data2 = pd.read_csv(path2)
            data3 = pd.read_csv(path3)
            number1 = data1.shape[0]
            print(f"{path1.split('/')[-1]} size: {number1}")

            data_merge = pd.merge(data1, data3, how='outer')
            number2 = data_merge.shape[0]
            print(f"data_merge size: {number2}")
            data_merge = data_merge.query("(time >= t_s) & (time <= t_e)")
            number3 = data_merge.shape[0]
            print(f"data_merge after time filter, size: {number3}", "\n")

            if number3 == number2 == number1:
                save_path = f"./数据集level{i}/航迹表/{file.replace('关联结果', '航迹')}"
                data_merge.to_csv(save_path, index=False)
            else:
                raise print("数据对不上")


def data_spliting():
    data = []
    big_data = []
    for i in [1, 2]:
        print(i)
        files = os.listdir(f"./数据集level{i}/航迹表/")
        for j, file in enumerate(files):
            path = f"./数据集level{i}/航迹表/{file}"
            df = pd.read_csv(path)
            df['level'] = i
            pattern = r"航迹-(\d+).csv"
            match = re.search(pattern, path)
            number = match.group(1)
            df['scene'] = number
            data.append(df)
            if data.__len__() > 1000:
                big_data.append(pd.concat(data))
                data = []

            print("\r", end="")
            print(f"file number: {j}/{len(files)}", end="")
            sys.stdout.flush()
        print("\r")

    print("----")
    if big_data:
        if big_data.__len__() > 1:
            data = pd.concat(big_data)
        else:
            data = big_data[0]
    else:
        data = pd.concat(data)
    del big_data
    data.drop(columns=["batch", "t_s", "t_e", "vel", "cou"], inplace=True)
    data["MMSI"] = data["mmsi"].map(lambda s: s[:9])
    data['level'] = data['level'].astype("category")
    data['scene'] = data['scene'].astype("category")
    data['source'] = data['source'].astype("category")

    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    data["userid"] = data.parallel_apply(lambda row: str(row.MMSI) + "-" + str(row.level) + "-" + str(row.scene),
                                         axis=1)
    data.drop(columns=["mmsi", "MMSI", "level", "scene"], inplace=True)
    print(data.info())

    print(f"source data size: {data.shape}")
    print(f"source data userid number: {len(data.userid.unique())}")

    data1 = data.query("source == 9001")
    data1 = data1[["userid", "time", "lat", "lon"]]
    data1.to_csv("9001.csv", index=False)
    print(f"9001 data size: {data1.shape}")
    print(f"9001 data userid number: {len(data1.userid.unique())}")

    data2 = data.query("source == 9002")
    data2 = data2[["userid", "time", "lat", "lon"]]
    data2.to_csv("9002.csv", index=False)
    print(f"9002 data size: {data2.shape}")
    print(f"9002 data userid number: {len(data2.userid.unique())}")


def data_dealing():
    data1 = pd.read_csv("9001.csv")
    print(f"9001 data size: {data1.shape}")
    print(f"9001 data userid number: {len(data1.userid.unique())}")

    data2 = pd.read_csv("9002.csv")
    print(f"9002 data size: {data2.shape}")
    print(f"9002 data userid number: {len(data2.userid.unique())}")

    userid1 = data1.userid.unique().tolist()
    userid2 = data2.userid.unique().tolist()
    common_user = list(set(userid1) & set(userid2))
    print(f"9001 and 9002 common user number: {common_user.__len__()}")
    data1 = data1.query(f"userid in {common_user}")
    data1.to_csv("9001_processed.csv", index=False)
    print(f"9001 data size: {data1.shape}")
    print(f"9001 data userid number: {len(data1.userid.unique())}")
    data2 = data2.query(f"userid in {common_user}")
    data2.to_csv("9002_processed.csv", index=False)
    print(f"9002 data size: {data2.shape}")
    print(f"9002 data userid number: {len(data2.userid.unique())}")

    # dic1 = data1.userid.value_counts()
    # dic2 = data2.userid.value_counts()
    # min_number = 10
    # filter_user1 = [k for k, v in dic1.items() if v < min_number]
    # print(f"9001 data filter userid number: {filter_user1.__len__()}")
    # filter_user2 = [k for k, v in dic2.items() if v < min_number]
    # print(f"9002 data filter userid number: {filter_user2.__len__()}")
    # filter_user = list(set(filter_user1 + filter_user2))
    # print(f"9001 and 9002 filter userid number: {filter_user.__len__()}")
    # data1 = data1.query(f"userid not in {filter_user}")
    # print(f"9001 data size: {data1.shape}")
    # print(f"9001 data userid number: {len(data1.userid.unique())}")
    # data2 = data2.query(f"userid not in {filter_user}")
    # print(f"9002 data size: {data2.shape}")
    # print(f"9002 data userid number: {len(data2.userid.unique())}")


def main():
    # data_merging()
    # data_spliting()
    data_dealing()


if __name__ == "__main__":
    main()
