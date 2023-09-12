import pandas as pd
from sklearn.metrics import confusion_matrix


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
