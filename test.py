if __name__ == "__main__":
    import pandas as pd

    # 创建多层次索引的DataFrame
    data = {'A': [1, 1, 2, 2], 'B': ['a', 'b', 'a', 'b'], 'C': [10, 20, 30, 40]}
    df = pd.DataFrame(data).set_index(['A', 'B'])
    print(df)

    print(df.reset_index(level="A"))
