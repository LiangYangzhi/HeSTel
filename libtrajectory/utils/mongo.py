import pandas as pd
from pymongo import MongoClient


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
