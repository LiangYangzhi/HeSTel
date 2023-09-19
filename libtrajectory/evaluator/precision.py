import pandas as pd


def evaluation(X_test, index, pred, config):
    pred = pred.reshape((X_test.shape[0], 1))
    pred = pd.DataFrame(data=pred, columns=['probably'])
    data = pd.merge(index, pred, how="outer", left_index=True, right_index=True)

    # Todo 评价标准化
    group_name = [config['preprocessing']['data1']['columns']['user'], 'segment']
    df_sort = data.sort_values(by='probably', ascending=False).groupby(group_name)
    positive_num = data[data["label"] == 1].shape[0]
    if positive_num == 0:
        raise print("Test dataset has no positive samples")

    # top1 precision
    df1: pd.DataFrame = df_sort.head(1)
    top1 = df1[df1["label"] == 1].shape[0]
    print(f"top5: {top1}/{positive_num}={top1 / positive_num}")

    # top5 precision
    df5: pd.DataFrame = df_sort.head(5)
    top5 = df5[df5["label"] == 1].shape[0] / positive_num
    print(f"top5: {top5}/{positive_num}={top5 / positive_num}")
    return top1, top5

