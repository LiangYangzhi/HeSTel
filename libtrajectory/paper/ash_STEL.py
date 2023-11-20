import json
import time
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import faiss
# from pandarallel import pandarallel


dir_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logs_path = f"./logs/ash_STEL/{dir_name}"
os.makedirs(logs_path)
t0 = datetime.now()


class Preprocessor(object):

    def __init__(self):
        self.result = None
        self.userid9002 = None
        self.userid9001 = None
        self.vector9002 = None
        self.vector9001 = None
        self.spatial_model = None
        self.split_cell = None
        self.data = None
        self.cell_latlon = {}
        self.max_point_num = 10000

    def loader(self):
        data9001 = pd.read_csv("../dataset/MTAD/9001.csv")
        data9001['userid'] = data9001['userid'].map(lambda u: u + '_9001')
        data9001.sort_values("time", inplace=True)
        print(f"data9001 number: {len(data9001.userid.unique())}")
        print(f"data9001 shape: {data9001.shape}")

        data9002 = pd.read_csv("../dataset/MTAD/9002.csv")
        data9002['userid'] = data9002['userid'].map(lambda u: u + '_9002')
        data9002.sort_values("time", inplace=True)
        print(f"data9002 number: {len(data9002.userid.unique())}")
        print(f"data9002 shape: {data9002.shape}")

        self.data = pd.concat([data9001, data9002])
        self.data.reset_index(drop=True, inplace=True)
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def _is_cell(self, lat, lon, cell_name):
        if cell_name not in self.split_cell:
            return cell_name

        for i in [1, 2, 3, 4]:
            cell_latlon = self.cell_latlon[f"{cell_name}{i}"]
            if (cell_latlon["min_lat"] <= lat <= cell_latlon["max_lat"] and
                    cell_latlon["min_lon"] <= lon <= cell_latlon["max_lon"]):
                return f"{cell_name}{i}"

        raise print(f"current cell is {cell_name}, \n cell_latlon: {self.cell_latlon}, \n "
                    f"current point (lat, lon) == {(lat, lon)} no find 4split cell")

    def _cell_4split(self, cell_latlon, cell_name):
        if not cell_latlon:
            cell_latlon = {"min_lat": self.data.lat.min(),
                           "min_lon": self.data.lon.min(),
                           "max_lat": self.data.lat.max(),
                           "max_lon": self.data.lon.max()}
            # cell = {"min_lat": -90, "min_lon": -180, "max_lat": 90, "max_lon": 180}

        median_lat = (cell_latlon['min_lat'] + cell_latlon['max_lat']) / 2
        median_lon = (cell_latlon['min_lon'] + cell_latlon['max_lon']) / 2

        self.cell_latlon[f'{cell_name}1'] = {"min_lat": median_lat,
                                             "min_lon": cell_latlon['min_lon'],
                                             "max_lat": cell_latlon['max_lat'],
                                             "max_lon": median_lon}

        self.cell_latlon[f'{cell_name}2'] = {"min_lat": median_lat,
                                             "min_lon": median_lon,
                                             "max_lat": cell_latlon['max_lat'],
                                             "max_lon": cell_latlon['max_lon']}

        self.cell_latlon[f'{cell_name}3'] = {"min_lat": cell_latlon['min_lat'],
                                             "min_lon": median_lon,
                                             "max_lat": median_lat,
                                             "max_lon": cell_latlon['max_lon']}

        self.cell_latlon[f'{cell_name}4'] = {"min_lat": cell_latlon['min_lat'],
                                             "min_lon": cell_latlon['min_lon'],
                                             "max_lat": median_lat,
                                             "max_lon": median_lon}

    def discretizing(self):
        self.data: pd.DataFrame
        # spatial discretization
        self.data['cell'] = ''
        print(f"data info : {self.data.info()}")
        print(f"cell point max number threshold: {self.max_point_num}\n")
        while True:
            # judge the points number of cell
            cell_points = self.data.cell.value_counts().to_dict()
            print(f"cell points number: {cell_points}")
            self.split_cell = [cell_name for cell_name, cell_num in cell_points.items()
                               if cell_num > self.max_point_num]
            if self.split_cell.__len__() == 0:
                break
            # generate split cells lat lon
            for cell_name in self.split_cell:
                cell_latlon = self.cell_latlon.get(cell_name, None)
                self._cell_4split(cell_latlon, cell_name)

            # update the cell name to which the point belongs
            # pandarallel.initialize(progress_bar=True)  parallel_apply
            self.data['cell'] = self.data.apply(lambda row: self._is_cell(row.lat, row.lon, row.cell), axis=1)
            print(f'Running time: {datetime.now() - t0} Seconds', '\n')

        print(f"cell len: {len(self.data.cell.unique())}, specific name: {self.data.cell.unique()}")
        self.data.to_csv(f"{logs_path}/discretize_{self.max_point_num}.csv", index=False)
        with open(f'cell_latlon_{self.max_point_num}.json', 'w') as f:
            json.dump(self.cell_latlon, f)

    def pretrain(self):
        self.data: pd.DataFrame
        # spatial
        corpus = self.data.groupby("userid").agg({"cell": list}).cell.tolist()
        model_config = {
            "min_count": 1,
            "workers": 10,
            "window": 5,
            "vector_size": 128,
            "epochs": 10,
            "sg": 1
        }
        self.spatial_model = Word2Vec(sentences=corpus, **model_config)
        self.spatial_model.save(f"{logs_path}/spatial_model_{self.max_point_num}.model")
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def build_vector(self):
        data9001 = self.data.query("userid.str.contains('_9001')", engine='python')
        data9001 = data9001.groupby("userid", as_index=False).agg({"cell": list})
        data9001['vector'] = data9001['cell'].map(
            lambda cell: np.mean([self.spatial_model.wv[i] for i in cell], axis=0))
        data9001['vector'] = data9001['vector'].map(lambda v: v.reshape(1, -1))
        self.vector9001 = np.concatenate(data9001['vector'], axis=0)
        self.userid9001 = {i: u for i, u in enumerate(data9001.userid.tolist())}

        data9002 = self.data.query("userid.str.contains('_9002')", engine='python')
        data9002 = data9002.groupby("userid", as_index=False).agg({"cell": list})
        data9002['vector'] = data9002['cell'].map(
            lambda cell: np.mean([self.spatial_model.wv[i] for i in cell], axis=0))
        data9002['vector'] = data9002['vector'].map(lambda v: v.reshape(1, -1))
        self.vector9002 = np.concatenate(data9002['vector'], axis=0)
        self.userid9002 = {i: u for i, u in enumerate(data9002.userid.tolist())}
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def _result_format(self, distances, indices, k):
        indices = pd.DataFrame(indices)
        indices.rename(columns={i: f"user9001_rank{i}" for i in range(k)}, inplace=True)
        print(f"indices shape: {indices.shape}")
        distances = pd.DataFrame(distances)
        distances.rename(columns={i: f"distance_rank{i}" for i in range(k)}, inplace=True)
        print(f"distances shape: {distances.shape}")
        result = indices.join(distances, how="inner")
        result['userid9002'] = result.index
        print(f"result(indices join distances) shape: {result.shape}")

        result["result"] = result.apply(
            lambda row: [{"userid9001": row[f"user9001_rank{i}"],
                          "distance": row[f"distance_rank{i}"],
                          "rank": i + 1} for i in range(k)], axis=1)
        result = result[['userid9002', 'result']]
        result = result.explode('result')
        print(f"result explode shape: {result.shape}")
        result.reset_index(drop=True, inplace=True)
        result = pd.concat([result, result['result'].apply(pd.Series)], axis=1).drop('result', axis=1)
        result['userid9002'] = result['userid9002'].map(self.userid9002)
        result['userid9001'] = result['userid9001'].map(self.userid9001)
        print(result.info())

        result['label'] = result.apply(lambda row: 1 if row.userid9001[:-2] == row.userid9002[:-2] else 0, axis=1)
        result.to_csv(f"{logs_path}/result_{self.max_point_num}.csv", index=False)
        self.result = result

    def query(self):
        k = 5
        admin = self.vector9001.shape[1]
        indexIP = faiss.IndexFlatIP(admin)
        indexIP.add(self.vector9001)
        distances, indices = indexIP.search(self.vector9002, k)
        self._result_format(distances, indices, k)
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def evaluator(self):
        user_num = len(self.result.userid9002.unique())
        top1_num = self.result.query("(label == 1) & (rank <= 1)").shape[0]
        print(f"top1: {top1_num}/{user_num} = {top1_num / user_num}")

        top3_num = self.result.query("(label == 1) & (rank <= 3)").shape[0]
        print(f"top3: {top3_num}/{user_num} = {top3_num / user_num}")

        top5_num = self.result.query("(label == 1) & (rank <= 5)").shape[0]
        print(f"top5: {top5_num}/{user_num} = {top5_num / user_num}")


def pipeline():
    preprocessor = Preprocessor()
    preprocessor.loader()
    preprocessor.discretizing()
    preprocessor.pretrain()
    preprocessor.build_vector()
    preprocessor.query()
    preprocessor.evaluator()

    if os.path.exists('./ash_STEL.logs'):
        shutil.copy('./ash_STEL.logs', f"{logs_path}/ash_STEL.logs")
    print(f'Running time: {datetime.now() - t0} Seconds', '\n')


if __name__ == "__main__":
    pipeline()
