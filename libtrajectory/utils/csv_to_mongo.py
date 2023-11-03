import json

import pandas as pd
import pymongo

if __name__ == "__main__":
    config = {
        "train": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
                  "db": "unify_face_lab",  # 库名
                  "table": "train_feature_lab"  # 表名
                  },
        "test": {"mongo": "mongodb://192.168.31.222:29017/admin",  # mongo连接
                 "db": "unify_face_lab",  # 库名
                 "table": "test_feature_lab"  # 表名
                 }
    }

    train = pd.read_csv('train_feature_7.csv')
    url = pymongo.MongoClient(config["train"]["mongo"])[config["train"]["db"]]
    coll = url[config["train"]["table"]]
    coll.insert_many(json.loads(train.T.to_json()).values())
    print("Done train")

    test = pd.read_csv('test_feature_3.csv')
    url = pymongo.MongoClient(config["test"]["mongo"])[config["test"]["db"]]
    coll = url[config["test"]["table"]]
    coll.insert_many(json.loads(test.T.to_json()).values())
    print("Done test")
