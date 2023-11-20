from functools import reduce

import pymongo
import lightgbm as lgb
import pandas as pd

config = {
    "label": {"mongo": "mongodb://192.168.31.149:27020/admin",  # mongo连接
              "db": "label",  # 库名
              "table": "face_imsi_zg"  # 表名
              },
    "train": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
              "db": "unify_face_lab",  # 库名
              "table": "trainingFeature"  # 表名
              },
    "test": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
             "db": "unify_face_lab",  # 库名
             "table": "testFeature"  # 表名
             }
}


def load_data():
    url = pymongo.MongoClient(config["label"]["mongo"])[config["label"]["db"]]
    coll = url[config["label"]["table"]]
    label = pd.DataFrame(coll.find({}))
    columns = ["fid", 'imsi']
    label = label[columns]
    label['imsi'] = label['imsi'].astype('object')
    label["label"] = 1
    print(label.info())

    columns = ['fid', 'imsi', 'taskId', "collTimeDiff", "sameSpace", 'cfSpaceCover', 'ciSpaceCover', 'cSameCount',
               'sspace_imsi', 'sspace_distMax', 'sspace_seqMax', 'cThreCount', 'cSpace_imsi', 'space_distMax',
               'space_seqMax', "cCollCount", "cfCount", "ciCount", "cSpace", "cfDen", "cfCoinDen", "cDiff",
               "new_sCollScore", "new_sSameScore", "new_sThreScore", "new_sfCoinDenScore", "new_sfSpaceCoverScore"]
    url = pymongo.MongoClient(config["train"]["mongo"])[config["train"]["db"]]
    coll = url[config["train"]["table"]]
    data = []
    train = []
    dataset = coll.find({})
    for document in dataset:
        data.append(document)
        if len(data) >= 10000:
            train.append(pd.DataFrame(data))
            data = []
    if data:
        train.append(pd.DataFrame(data))
    train = pd.concat(train)
    train = train[columns]

    url = pymongo.MongoClient(config["test"]["mongo"])[config["train"]["db"]]
    coll = url[config["test"]["table"]]
    data = []
    test = []
    dataset = coll.find({})
    for document in dataset:
        data.append(document)
        if len(data) >= 10000:
            test.append(pd.DataFrame(data))
            data = []
    if data:
        test.append(pd.DataFrame(data))
    test = pd.concat(test)
    test = test[columns]

    print('-' * 30)
    print(train.fid.unique().tolist().__len__())
    train = train.merge(label, how="left")
    train["label"].fillna(0, inplace=True)
    fid = train[train['label'] == 1].fid.unique().tolist()
    train = train[train["fid"].isin(fid)]
    print(fid.__len__())
    print('-' * 30)
    y_train = train.label.to_list()
    X_train = train.drop(columns=['fid', 'imsi', 'taskId', 'label'])

    test = test.merge(label, how="left")
    test["label"].fillna(0, inplace=True)
    y_test = test.label.to_list()
    index = test[['fid', 'imsi', 'taskId']]
    index.reset_index(drop=True, inplace=True)
    X_test = test.drop(columns=['fid', 'imsi', 'taskId', 'label'])

    return X_train, X_test, y_train, y_test, index


def predict(model, X_test, y_test, index, group_name):
    pred = model.predict(X_test).reshape((X_test.shape[0], 1))
    pred = pd.DataFrame(data=pred, columns=['probably'])
    df = pd.DataFrame(data=y_test, columns=['label'])
    df = pd.merge(pred, df, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, index, how='outer', left_index=True, right_index=True)

    positive_num = df[df["label"] == 1].shape[0]
    df_sort = df.sort_values(by='probably', ascending=False).groupby(group_name)

    df1 = df_sort.head(1)
    top1 = df1[df1["label"] == 1].shape[0]

    df5 = df_sort.head(5)
    top5 = df5[df5["label"] == 1].shape[0]
    print(f'top1: {top1}/{positive_num}={top1 / positive_num}, top5: {top5}/{positive_num}={top5 / positive_num}')


def main():
    X_train, X_test, y_train, y_test, index = load_data()
    clf = lgb
    train_matrix = clf.Dataset(X_train, label=y_train)
    param = {'objective': 'binary'}
    model = clf.train(param, train_matrix)
    predict(model, X_test, y_test, index, ['fid', 'taskId'])


if __name__ == "__main__":
    main()
