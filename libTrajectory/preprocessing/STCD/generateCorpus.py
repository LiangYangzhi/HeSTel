import copy
import math
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from transbigdata import getdistance


class Generator(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def user_place(self, sort_col='timestamp', id_col="userid", target_col='placeid',
                   data_aug=None, aug_min_length=10):
        data = self.data.sort_values(sort_col)
        userid_list = data[id_col].unique().tolist()
        corpus = []

        if isinstance(target_col, list):
            data['word'] = ''
            for c in target_col:
                data['word'] = data['word'] + '-' + data[c].astype(str)
        else:
            data['word'] = data[target_col]

        data_group = data.groupby(id_col)
        print('--getting corpus...')

        for userid in userid_list:
            group = data_group.get_group(userid)
            tra = group['word'].tolist()
            corpus.append(tra)
            if data_aug and tra.__len__() >= aug_min_length:
                for _ in range(data_aug):
                    i = random.randint(2, len(tra) - 2)
                    tra1 = tra[0:i]
                    tra2 = tra[i + 1:]
                    tra3 = tra1 + tra2
                    corpus.extend([tra1, tra2, tra3])
        return corpus

    def place_lon_lat(self, potential_nerghbor=50, walk_num=10, walk_len=128):
        data = self.data[['placeid', 'lat', 'lon']]
        data.drop_duplicates('placeid', inplace=True)
        placeid_list = data['placeid'].unique().tolist()
        data.set_index(keys='placeid', inplace=True)

        print('getting coordinates...')
        coord = [[data.loc[i]['lat'], data.loc[i]['lon']] for i in placeid_list]
        coord = np.array(coord)
        tree = KDTree(coord, leaf_size=potential_nerghbor)
        print('KD tree constructed')

        print('getting neighbor distance...')
        distance_dict = {}
        for id1 in placeid_list:
            distance_dict[id1] = {}
            lat = data.loc[id1]['lat']
            lon = data.loc[id1]['lon']
            dist, ind = tree.query(np.array([lat, lon]).reshape(1, -1), k=potential_nerghbor)
            for i in range(len(ind[0])):
                id2 = placeid_list[ind[0][i]]
                if id2 != id1:
                    distance_dict[id1][id2] = self.distance2weight(dist[0][i])

        print('generateing random walk paths...')
        corpus = []
        d = {'distance': [], 'id': [], 'num': []}
        for id1 in placeid_list:
            for i in range(walk_num):
                current = id1
                walk_step = 0
                trajectory = [copy.deepcopy(current)]
                while walk_step < walk_len:
                    current = random.choices(population=list(distance_dict[current].keys()),
                                             weights=list(distance_dict[current].values()))[0]
                    walk_step = walk_step + 1
                    trajectory.append(copy.deepcopy(current))
                corpus.append(copy.deepcopy(trajectory))
                d['id'].append(id1)
                d['num'].append(i)
                id2 = trajectory[-1]
                lat1 = data.loc[id1]['lat']
                lon1 = data.loc[id1]['lon']
                lat2 = data.loc[id2]['lat']
                lon2 = data.loc[id2]['lon']
                d['distance'].append(getdistance(lon1, lat1, lon2, lat2))
        print('random walk complete')
        d = pd.DataFrame(d)
        print('random walk info:')
        print(d['distance'].describe())
        return corpus

    def distance2weight(self, d):
        return math.exp(-d * d)
