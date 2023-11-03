import os
import re

import pandas as pd


def main():
    # data1 = pd.read_csv("./数据集level1/量测场景/场景-0.csv")
    # data2 = pd.read_csv("./数据集level1/真实场景/场景-0.csv")
    # data3 = pd.read_csv("./数据集level1/关联表/关联结果-0.csv")
    #
    # data_merge2 = pd.merge(data1, data3, how='outer')
    # print(data_merge2.info())
    # data_merge2 = data_merge2.query("(time >= t_s) & (time <= t_e)")
    # print(data_merge2.info())

    # 获取当前目录下的所有文件和文件夹
    files = os.listdir("./数据集level2/关联表/")
    for file in files:
        path1 = f"./数据集level2/量测场景/{file.replace('关联结果', '场景')}"
        # path2 = f"./数据集level1/真实场景/{file.replace('关联结果', '场景')}"
        path3 = f"./数据集level2/关联表/{file}"
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
            save_path = f"./数据集level2/航迹表/{file.replace('关联结果', '航迹')}"
            data_merge.to_csv(save_path, index=False)
        else:
            raise print("数据对不上")


if __name__ == "__main__":
    main()
