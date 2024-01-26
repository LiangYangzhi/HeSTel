import math

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from transbigdata import getdistance


class Generator(object):
    def __init__(self, data: pd.DataFrame):
        self.temporal_edge = None
        self.spatial_edge = None
        self.node = None
        self.data = data

    def hypergraph(self, t_thre, d_thre, topn):
        self.node = self.data['placeid'].unique().tolist()
        self.spatial_edge = self._generate_spatial_edge(topn, d_thre)
        self.temporal_edge = self._generate_temporal_edge(t_thre)

    def _generate_spatial_edge(self, topn, d_thre):
        placeid_df = self.data[['placeid', 'lat', 'lon']]
        placeid_df.drop_duplicates(subset='placeid', inplace=True)

        placeid_list = placeid_df['placeid'].unique().tolist()
        placeid_df.set_index(keys='placeid', inplace=True)
        coord = [[placeid_df.loc[id1]['lat'], placeid_df.loc[id1]['lon']] for id1 in placeid_list]
        coord = np.array(coord)
        tree = KDTree(coord, leaf_size=topn)

        ans = {}
        for p1 in placeid_list:
            lat1 = placeid_df.loc[p1]['lat']
            lon1 = placeid_df.loc[p1]['lon']
            _, ind = tree.query(np.array([lat1. lon1]).reshape(1, -1), k=topn+1)
            for i in range(len(ind[0])):
                p2 = placeid_list[ind[0][i]]
                if p2 != p1:
                    lat2 = placeid_df.loc[p2]['lat']
                    lon2 = placeid_df.loc[p2]['lon']
                    d = getdistance(lon1, lat1, lon2, lat2)
                    if d <= d_thre:
                        if p1 not in ans:
                            ans[p1] = {}
                        ans[p1][p2] = self._distance2weight(d)
        return ans

    def _distance2weight(self, d):
        return 9 * math.exp(-d * d) + 1

    def _generate_temporal_edge(self, t_thre):
        userid_list = self.data['userid'].unique().tolist()
        data_group = self.data.groupby("userid")
        temporal_edge = {}
        for user in userid_list:
            user_data = data_group.get_group(user)
            for i, row in enumerate(user_data.iterrows()):
                if i + 1 < len(user_data):
                    t1 = user_data.iloc[i]['timestamp']
                    p1 = user_data.iloc[i]['placeid']
                    t2 = user_data.iloc[i + 1]['timestamp']
                    p2 = user_data.iloc[i]['placeid']
                    t_delta = abs(t2 - t1)
                    if t_delta <= t_thre:
                        if p1 not in temporal_edge:
                            temporal_edge[p1] = {}

                        if p2 not in temporal_edge[p1]:
                            temporal_edge[p1][p2] = 1
                        else:
                            temporal_edge[p1][p2] += 1
        return temporal_edge

    def get_data(self):
        return self.node, self.spatial_edge, self.temporal_edge
