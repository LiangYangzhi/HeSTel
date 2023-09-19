from geopy.distance import geodesic
import pandas as pd

if __name__ == "__main__":
    data = {'class': ["1班", "1班", "2班", "2班", "2班"],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'Age': [18, 19, 20, 21, 22]}
    df = pd.DataFrame(data)
    print(df.to_dict(orient="list"))  # {'dict', 'list', 'series', 'split', 'records', 'index'}

    # from progress.bar import Bar
    # import time
    #
    # # 创建Bar类的实例
    # bar = Bar('MyProcess:', max=100)
    # # 循环处理某业务，调用bar对象的next()方法，循环次数等于max
    # for _ in range(100):
    #     # Do some work
    #     time.sleep(0.05)
    #     bar.next()
    # # 循环完成后调用finish()方法
    # bar.finish()
    #
    # data = {'ID': [1, 2, 3, 4, 5],
    #         'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
    #         'Age': [18, 19, 20, 21, 22]}
    # df = pd.DataFrame(data)
    #
    # # 设置Hash列为索引
    # df.set_index('ID', inplace=True)
    # df['Name'] = df['Name'].astype('category')
    # print(df)

    # # 创建多层次索引的DataFrame
    # data = {'A': [1, 2], 'B': [[{'a': 1}, {'a': 2}, {'a': 3}], [{'b': 1}, {'b': 2}, {'b': 3}]], 'C': [10, 20]}
    # df = pd.DataFrame(data)
    # df = df.explode('B')
    # df = df.reset_index(drop=True)
    # # print(df)
    # # # {'dict', 'list', 'series', 'split', 'records', 'index'}
    # # print(df.to_dict('list'))
    # df["A"] = df["A"].astype("str")
    # print(df.info())
    #
    # print(df.reset_index(level="A"))
    #
    # print(df.index.get_level_values("A").unique().__len__())
    # print(df.to_dict(orient="list"))

    # p1 = (29.281932, 104.880884)
    # p2 = (29.280443, 104.882478)
    # print(geodesic(p1, (29.28093303, 104.8820641)).m)




