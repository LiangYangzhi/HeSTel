import json
from functools import reduce

import numpy as np
import pandas as pd
from pymongo import MongoClient

if __name__ == "__main__":
    config = {
        "feature": {"mongo": "mongodb://192.168.31.173:27017/admin",  # mongo连接
                    "db": "unify_face",  # 库名
                    "table": "featureDTO"  # 表名
                    },
        "label": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
                  "db": "unify_face_lab",  # 库名
                  "table": "lyz_label"  # 表名
                  }
    }


def mongo_to_df(client: str, db: str, collection: str, condition=None, columns=None):
    """
    load mongo dataset in pd.DataFrame model.
    :param client: mongo client
    :param db: mongo db
    :param collection: mongo collection
    :param condition: dict or None
    :param columns: list or None
    :return: pd.DataFrame
    """
    if condition is None:
        condition = {}
    url = MongoClient(client)[db]
    coll = url[collection]
    cursor = coll.find(condition)
    df = []
    data = []
    count = 0  # max records
    for document in cursor:
        data.append(document)
        count += 1
        if count >= 100000:
            sub = pd.DataFrame(data)
            if columns:
                sub = sub[columns]
            df.append(sub)
            print(f"{len(df) * 100000}")
            data = []
            count = 0
    if count:
        sub = pd.DataFrame(data)
        if columns:
            sub = sub[columns]
        df.append(sub)
    if len(df) == 0:
        df = pd.DataFrame()
    else:
        df = pd.concat(df)
    return df


def main():
    client = config['feature']['mongo']
    db = config['feature']['db']
    collection = config['feature']['table']
    condition = {}
    columns = ["fid", "imsi", "score", "allCollTimeDiff", "cSpace_imsi", "sspace_imsi"]
    df = mongo_to_df(client, db, collection, condition, columns)
    print(df.info())
    print(f"fid number: {len(df.fid.unique())}")

    allCollTimeDiff_fid = df.query(f"allCollTimeDiff >= {14 * 24 * 60 * 60}")['fid'].unique().tolist()
    print(f"after allCollTimeDiff filter, fid number: {allCollTimeDiff_fid.__len__()}")

    df['space_imsi'] = df.apply(lambda row: row.sspace_imsi + row.cSpace_imsi, axis=1)
    space_imsi_fid = df.query("space_imsi >= 5")['fid'].unique().tolist()
    print(f"after sspace_imsi filter, fid number: {space_imsi_fid.__len__()}")

    score_fid = df.query("score >= 0.8")['fid'].unique().tolist()
    print(f"after score filter, fid number: {score_fid.__len__()}")

    df["fid_rank"] = df.groupby("fid")['score'].rank(ascending=False)
    df['imsi_rank'] = df.groupby("imsi")['score'].rank(ascending=False)
    double_fid = df.query("(fid_rank == 1) & (imsi_rank == 1)")['fid'].unique().tolist()
    print(f"after double top1 filter, fid number: {double_fid.__len__()}")

    fid_list = list(reduce(lambda x, y: set(x) & set(y),
                      [allCollTimeDiff_fid, space_imsi_fid, score_fid, double_fid]))
    print(f"Simultaneously meeting the above four conditions, fid number: {fid_list.__len__()}")

    df = df[df["fid"].isin(fid_list)]
    group = df.groupby("fid")
    distance = []
    for fid in fid_list:
        user = group.get_group(fid)
        top1_score = user[user['fid_rank'] == 1]['score'].tolist()
        top2_score = user[user['fid_rank'] == 2]['score'].tolist()
        if top1_score.__len__() == 1 and top2_score.__len__() == 1:
            if top1_score[0] - top2_score[0] >= 0.1:
                distance.append(fid)

    print(f"top1 - top2 distance condition, fid number: {distance.__len__()}")
    df_top1 = df.query("fid_rank == 1")
    df_top1 = df_top1[df_top1['fid'].isin(distance)]
    label = df_top1[["fid", "imsi"]]
    print(label.shape)

    label.reset_index(drop=True, inplace=True)
    url = MongoClient(config["label"]["mongo"])[config["label"]["db"]]
    coll = url[config["label"]["table"]]
    coll.insert_many(json.loads(label.T.to_json()).values())


if __name__ == "__main__":
    main()
