import json
from functools import reduce

import numpy as np
import pandas as pd
from pymongo import MongoClient

if __name__ == "__main__":
    config = {
        "feature": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
                    "db": "unify_face_lab",  # 库名
                    "table": "trainingFeature"  # 表名
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


def load_matrix(name):
    with open(f'./matrix/{name}.json', 'r') as f:
        content = f.read()
    matrix = json.loads(content)
    return np.array(matrix)


def same_trunk_ratio(sspace_distMax, collTimeDiff):
    if sspace_distMax >= 5000 or collTimeDiff >= 3 * 24 * 60 * 60:
        ratio = 1
    elif sspace_distMax >= 3000 or collTimeDiff >= 2 * 24 * 60 * 60:
        ratio = 0.8
    else:
        ratio = 0.6
    return ratio


def same_score(matrix_name, sameTrunkRatio, cSameCount, sameSpace):
    if cSameCount < 1:
        return 0

    matrix = load_matrix(matrix_name)
    i_max, j_max = matrix.shape  # i is row space, j is column collCount
    row = i_max - 1 if int(sameSpace) >= i_max else int(sameSpace - 1)
    column = j_max - 1 if int(cSameCount) >= j_max else int(cSameCount - 1)
    score = matrix[row][column]

    return score * sameTrunkRatio


def general_trunk_ratio(space_distMax, collTimeDiff):
    if space_distMax >= 5000 or collTimeDiff >= 3 * 24 * 60 * 60:
        ratio = 1
    elif space_distMax >= 3000 or collTimeDiff >= 2 * 24 * 60 * 60:
        ratio = 0.8
    else:
        ratio = 0.6
    return ratio


def general_score(matrix_name, threTrunkRatio, cThreCount, cSpace):
    if cThreCount < 1:
        return 0

    matrix = load_matrix(matrix_name)
    i_max, j_max = matrix.shape  # i is row space, j is column collCount
    row = i_max - 1 if int(cSpace) >= i_max else int(cSpace - 1)
    column = j_max - 1 if int(cThreCount) >= j_max else int(cThreCount - 1)
    score = matrix[row][column]

    return score * threTrunkRatio


def cool_score(name, sameScore, threScore):
    if name == "high":
        return 0.2 * sameScore + 0.8 * threScore
    elif name == "middle":
        return 0.4 * sameScore + 0.6 * threScore
    elif name == "low":
        return 0.9 * sameScore + 0.1 * threScore
    else:
        return -1


def main():
    client = config['feature']['mongo']
    db = config['feature']['db']
    collection = config['feature']['table']
    condition = {}
    columns = ["fid", "imsi", "allCollTimeDiff", "collTimeDiff", "sspace_imsi", "cSpace_imsi",
               "cSameCount", "cThreCount", "sspace_distMax", "space_distMax"]
    df = mongo_to_df(client, db, collection, condition, columns)
    print(df.info())

    name = 'high'
    # df['score'] = df.apply(lambda row: row['sspace_imsi'] * 2 + row['cSpace_imsi'], axis=1)
    df["sameTrunkRatio"] = df.apply(
        lambda row: same_trunk_ratio(row.sspace_distMax, row.collTimeDiff), axis=1)
    df["sameScore"] = df.apply(
        lambda row: same_score(f"{name}Same", row.sameTrunkRatio, row.cSameCount, row.sspace_imsi), axis=1)
    df["threTrunkRatio"] = df.apply(
        lambda row: general_trunk_ratio(row.space_distMax, row.collTimeDiff), axis=1)
    df["threScore"] = df.apply(
        lambda row: general_score(f"{name}Thre", row.threTrunkRatio, row.cThreCount, row.cSpace_imsi), axis=1)
    df["score"] = df.apply(lambda row: cool_score(name, row.sameScore, row.threScore), axis=1)

    columns = ["fid", "imsi", "allCollTimeDiff", "sspace_imsi", "cSpace_imsi", "score"]
    df = df[columns]
    print(f"fid number: {len(df.fid.unique())}")

    allCollTimeDiff_fid = df.query(f"allCollTimeDiff >= {7 * 24 * 60 * 60}")['fid'].unique().tolist()
    print(f"after allCollTimeDiff filter, fid number: {allCollTimeDiff_fid.__len__()}")

    df['space_imsi'] = df.apply(lambda row: row.sspace_imsi + row.cSpace_imsi, axis=1)
    space_imsi_fid = df.query("space_imsi >= 5")['fid'].unique().tolist()
    print(f"after sspace_imsi filter, fid number: {space_imsi_fid.__len__()}")
    # sspace_imsi_fid = df.query("sspace_imsi >= 2")['fid'].unique().tolist()
    # print(f"after sspace_imsi filter, fid number: {sspace_imsi_fid.__len__()}")
    #
    # cSpace_imsi_fid = df.query("cSpace_imsi >= 3")['fid'].unique().tolist()
    # print(f"after cSpace_imsi filter, fid number: {cSpace_imsi_fid.__len__()}")

    df["fid_rank"] = df.groupby("fid")['score'].rank(ascending=False)
    df['imsi_rank'] = df.groupby("imsi")['score'].rank(ascending=False)
    double_fid = df.query("(fid_rank == 1) & (imsi_rank == 1)")['fid'].unique().tolist()
    print(f"after double top1 filter, fid number: {double_fid.__len__()}")

    fid_list = list(reduce(lambda x, y: set(x) & set(y),
                      [allCollTimeDiff_fid, space_imsi_fid, double_fid]))
    print(f"Simultaneously meeting the above four conditions, fid number: {fid_list.__len__()}")

    df = df[df["fid"].isin(fid_list)]
    group = df.groupby("fid")
    distance = []
    for fid in fid_list:
        user = group.get_group(fid)
        top1_score = user[user['fid_rank'] == 1]['score'].tolist()
        top2_score = user[user['fid_rank'] == 2]['score'].tolist()
        if top1_score.__len__() == 1 and top2_score.__len__() == 1:
            if top1_score[0] - top2_score[0] >= 0.2:
                distance.append(fid)

    print(f"top1 - top2 distance condition, fid number: {distance.__len__()}")
    df_top1 = df.query("fid_rank == 1")
    df_top1 = df_top1[df_top1['fid'].isin(distance)]
    print(df_top1.info())

    url = MongoClient(config["label"]["mongo"])[config["label"]["db"]]
    coll = url[config["label"]["table"]]
    coll.insert_many(json.loads(df_top1.T.to_json()).values())


if __name__ == "__main__":
    main()
