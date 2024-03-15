import logging
from random import sample

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from libTrajectory.evaluator.faiss_cosine import evaluator


class Pretrain(object):
    def __init__(self, data_path):
        self.data_path = data_path
        logging.info(f"self.path={self.data_path}")

    def loader(self):
        logging.info("data loading...")
        dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'spaceid': str, 'timeid': str, 'stid': str}
        logging.info("generate graph1...")
        logging.info("data preparation...")
        data1 = pd.read_csv(f"{self.data_path[: -4]}_data1.csv", dtype=dic)
        data1 = data1[['tid', 'time', 'stid']].copy()
        data1['tid'] = data1['tid'].map(lambda x: f"{x}_1")
        data2 = pd.read_csv(f"{self.data_path[: -4]}_data2.csv", dtype=dic)
        data2 = data2[['tid', 'time', 'stid']].copy()
        data2['tid'] = data2['tid'].map(lambda x: f"{x}_2")
        data = pd.concat([data1, data2])
        data.sort_values(['time'], inplace=True)
        data = data[['tid', 'stid']]
        self.data = data

    def generate_corpus(self):
        logging.info("generate corpus...")
        corpus = []
        group = self.data.groupby('tid')
        tid_list1 = self.data.tid.unique().tolist()
        tid_list2 = self.data.tid.unique().tolist()
        tid_list = tid_list1 + tid_list2
        for tid in tid_list:
            df = group.get_group(tid)
            tra = df['stid'].tolist()
            corpus.append(tra)
            if len(set(tra)) > 6 and len(set(tra)) != len(tra):
                i = tra[0]
                tra1 = [i]
                for j in tra:
                    if i != j:
                        tra1.append(j)
                        i = j
                corpus.append(tra1)

                stid = list(set(tra))
                stid.remove(tra[0])
                stid.remove(tra[-1])
                random_remove = sample(stid, 1)[0]
                tra2 = [i for i in tra if i != random_remove]
                corpus.append(tra2)

        self.corpus = corpus
        logging.info("generate corpus completed")

    def run(self):
        self.loader()
        self.generate_corpus()
        para = {
            "min_count": 1,
            "workers": 24,
            "window": 5,
            "vector_size": 128,
            "epochs": 10,
            "sg": 1
        }
        logging.info(f"pretrain..., parameter={para}")
        model = Word2Vec(sentences=self.corpus, **para)
        model.save(f"{self.data_path[: -4]}_stid_model.model")
        logging.info("model train completed")
        return model

    def eval(self, test_path):
        logging.info("model eval...")
        dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'spaceid': str, 'timeid': str, 'stid': str}
        model = Word2Vec.load(f"{self.data_path[: -4]}_stid_model.model")
        for k, v in test_path.items():
            data1 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data1.csv", dtype=dic)
            data1 = data1[['tid', 'time', 'stid']].copy()
            data1.sort_values(['time'], inplace=True)
            data1 = data1[['tid', 'stid']].copy()
            group1 = data1.groupby('tid')

            data2 = pd.read_csv(f"{self.data_path[: -4]}_{k}_data2.csv", dtype=dic)
            data2 = data2[['tid', 'time', 'stid']].copy()
            data2.sort_values(['time'], inplace=True)
            data2 = data2[['tid', 'stid']].copy()
            group2 = data2.groupby('tid')

            tid_list = data1.tid.unique().tolist()
            embedding_1 = []
            embedding_2 = []
            for tid in tid_list:
                df1 = group1.get_group(tid)
                tra1 = df1['stid'].tolist()
                vec1 = [model.wv[i] for i in tra1]
                vec1 = np.array(vec1)
                vec1 = vec1.mean(axis=0, keepdims=True)[0]
                embedding_1.append(vec1)

                df2 = group2.get_group(tid)
                tra2 = df2['stid'].tolist()
                vec2 = [model.wv[i] for i in tra2]
                vec2 = np.array(vec2)
                vec2 = vec2.mean(axis=0, keepdims=True)[0]
                embedding_2.append(vec2)

            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
            evaluator(embedding_1, embedding_2)
        logging.info("model eval completed")

    def get(self, method="load"):
        if method == "load":
            model = Word2Vec.load(f"{self.data_path[: -4]}_stid_model.model")
            return model
        if method == "run":
            return self.run()
