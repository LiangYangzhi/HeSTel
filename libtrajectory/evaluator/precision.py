import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluation(X_test, index, pred, config):
    pred = pred.reshape((X_test.shape[0], 1))
    pred = pd.DataFrame(data=pred, columns=['probably'])
    data = pd.merge(index, pred, how="outer", left_index=True, right_index=True)

    # Todo 评价标准化
    group_name = [config['preprocessing']['data1']['columns']['user'], 'segment']
    df_sort = data.sort_values(by='probably', ascending=False).groupby(group_name)
    # top1 precision
    df1: pd.DataFrame = df_sort.head(1)
    df1.insert(0, column='pred', value=1)
    precision1 = precision(df1, ["label", "pred"])
    print(f"top1: {precision1}")

    # top5 precision
    df5: pd.DataFrame = df_sort.head(5)
    df5.insert(0, column='pred', value=1)
    precision5 = precision(df5, ["label", "pred"])
    print(f"top5: {precision5}")


def precision(data: pd.DataFrame, col: list):
    """
    precision = TP / (TP + FP)
    :param data:
    :param col: [label, predict]
    :return:
    """
    cm = confusion_matrix(data[col[0]], data[col[1]])
    tp = cm[1, 1]
    fp = cm[0, 1]
    precision_score = tp / (tp + fp)
    return precision_score
