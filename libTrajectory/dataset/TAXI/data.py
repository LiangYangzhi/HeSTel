import math
import os
import random
from datetime import datetime
from geopy.distance import geodesic
import numpy as np
import pandas as pd


class ProTra(object):
    def __init__(self):
        pass

    def main(self):
        self.car()
        self.traj_exp()
        self.traj()
        self.denoising()
        files = os.listdir(f'./traj')
        files = pd.DataFrame(data=files, columns=['f'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=24)
        files['tra'] = files['f'].parallel_map(lambda f: pd.read_csv(f'./traj/{f}', dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float}))
        df = pd.concat(files.tra.to_list())
        df.reset_index(drop=True, inplace=True)
        print(df.info())
        print(df.lat.min(), df.lat.max())
        print(df.lon.min(), df.lon.max())
        print(df.time.min(), df.time.max())   # 1201930244 1202463559
        self.min_time = df.time.min()
        self.max_time = df.time.max()
        self.multi_traj()
        self.test_data()
        self.split()

    def car(self):
        dir_path = f'./car'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        files = os.listdir('./original')
        colunms = ['uid', 'time', 'lon', 'lat']
        for f in files:
            if '.txt' not in f:
                print(f)
                continue
            df = pd.read_csv(f'./original/{f}', sep=',', header=None, names=colunms)
            df['time'] = df["time"].map(lambda t_str: datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S').timestamp())
            df.sort_values(by=['uid', 'time'], inplace=True)
            df.drop_duplicates(inplace=True, ignore_index=True)
            df = df.query(f"lat > 39 and lat < 42 and lon > 115 and lon < 118")
            if df.shape[0]:
                name = f.replace("txt", "csv")
                df.to_csv(f'{dir_path}/{name}')

    def traj_exp(self):
        files = os.listdir('./car')
        diff = []
        for f in files:
            df = pd.read_csv(f'./car/{f}')
            if df.shape[0] < 2:
                continue
            df['uid'] = df['uid'].astype("str")
            df['uid'] = df['uid'].astype("str")
            df.sort_values(['time'], inplace=True)
            df['diff'] = df['time'].diff()
            df['diff'].fillna(0, inplace=True)
            sub_diff = sum(df['diff'].tolist()) / (df.shape[0] - 1)
            diff.append(sub_diff)
        diff = np.array(diff)
        print(f"mean: {np.mean(diff)}")  # 3625s
        print(f"median: {np.median(diff)}")  # 404s

    def traj(self):
        dir_path = f'./traj'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        files = os.listdir('./car')
        inter = 2
        for f in files:
            df = pd.read_csv(f'./car/{f}')
            df['uid'] = df['uid'].astype("str")
            df['uid'] = df['uid'].astype("str")
            df.sort_values(['time'], inplace=True)
            df['diff'] = df['time'].diff()
            df['diff'].fillna(0, inplace=True)
            df.reset_index(drop=True, inplace=True)
            split_index = df.query(f"diff >= {60 * 60 * inter}").index.tolist()

            if len(split_index) == 0:
                df.drop(columns=['diff'], inplace=True)
                df['tid'] = df["uid"]
                df.to_csv(f'{dir_path}/{f}', index=False)
                continue

            df_list = []
            start_i = 0
            end_i = 0
            split_index.append(df.index[-1])
            for site, i in enumerate(split_index):
                end_i = i
                sub_df = df.loc[start_i: end_i]
                start_i = i + 1
                sub_df["tid"] = sub_df["uid"].map(lambda u: f"{u}_{site + 1}")
                time_shift =  week
                shift = random.sample(time_shift, 1)[0] * 24 * 60 * 60
                sub_df['time'] = sub_df['time'].map(lambda t: t+shift)
                df_list.append(sub_df)
            tracks = pd.concat(df_list)
            tracks.drop(columns=['diff'], inplace=True)
            tracks.to_csv(f'./traj/{f}', index=False)

    def denoising(self):
        files = os.listdir('./traj')
        thre = 100  # m
        for f in files:
            data = pd.read_csv(f'./traj/{f}', dtype={'uid': str, 'tid': str})
            tid_list = data.tid.unique().tolist()
            result = []
            for tid in tid_list:
                df = data.query(f"tid == '{tid}'")
                df.sort_values(['time'], inplace=True)
                df['latlon'] = df.apply(lambda row: (row.lat, row.lon), axis=1)
                if df.shape[0] <= 1:
                    df['flag'] = 1
                    result.append(df)
                    continue
                latlon = df.latlon.tolist()
                length = len(latlon)
                point = 0
                j = 0
                flag = [1]
                for i in range(1, length):
                    dis = geodesic(latlon[j], latlon[i]).m
                    if dis > thre:
                        point += 1
                        j = i
                        flag.append(1)  # 该轨迹点保留
                    else:
                        flag.append(0)  # 该轨迹点去除
                if flag.count(1) == 1:  # 至少保留两个轨迹点
                    flag[-1] = 1
                df['flag'] = flag
                df = df.query("flag == 1")
                result.append(df)

            result = pd.concat(result)
            result.drop(columns=['flag', 'latlon'], inplace=True)
            if result.shape[0] == 0:
                print(f"false uid: {f}, shape[0]==0")
            result.to_csv(f"./traj/{f}", index=False)

    def _random(self, tid, t, m_t):
        if tid in self.no_tid:
            return t + random.randint(-self.time, self.time)
        return m_t

    def _multi(self, lat, lon, t):
        # 该latlon落入那个区域内
        lati = None
        lat0 = self.lat_lis[0]
        for i, lat1 in enumerate(self.lat_lis[1:]):
            if lat0 <= lat <= lat1:
                lati = i
                break
            lat0 = lat1

        lonj = None
        lon0 = self.lon_lis[0]
        for j, lon1 in enumerate(self.lon_lis[1:]):
            if lon0 <= lon <= lon1:
                lonj = j
                break
            lon0 = lon1

        center = self.lat_lon[f"{lati}-{lonj}"]
        distance = int(geodesic(center, (lat, lon)).m)
        if random.randint(0, distance) <= self.space:
            multi_t = t + random.randint(-self.time, self.time)
            if multi_t < self.min_time:
                multi_t = self.min_time
            if multi_t > self.max_time:
                multi_t = self.max_time
            return {"m_time": multi_t, "m_lat": center[0], "m_lon": center[1], "m_did": f"{lati}-{lonj}"}
        return {"m_time": 0, "m_lat": center[0], "m_lon": center[1], "m_did": f"{lati}-{lonj}"}

    def multi_traj(self):
        files = os.listdir(f'./traj')
        files = pd.DataFrame(data=files, columns=['f'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=24, progress_bar=False)
        files['tra'] = files['f'].parallel_map(lambda f: pd.read_csv(f'./traj/{f}', dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float}))
        df = pd.concat(files.tra.to_list())
        df.reset_index(drop=True, inplace=True)
        print(df.info())

        lat_size, lon_size = 100, 100
        print(f"lat_size: {lat_size}, lon_size: {lon_size}")
        lat0, lat1 = df.lat.min(), df.lat.max()
        lon0, lon1 = df.lon.min(), df.lon.max()
        print(f"lat0: {lat0}, lat1: {lat1}, lon0: {lon0}, lon1: {lon1}")
        lat_len, lon_len = abs(lat1 - lat0), abs(lon1 - lon0)
        lat_step, lon_step = round(lat_len / lat_size, 5), round(lon_len / lon_size, 5)
        print(f"lat_len: {lat_len}, lon_len： {lon_len}， lat_step: {lat_step}, lon_step: {lon_step}")

        lat_lis, lon_lis = [lat0], [lon0]
        for _ in range(lat_size):
            lat_lis.append(round(lat_lis[-1] + lat_step, 5))
        for _ in range(lon_size):
            lon_lis.append(round(lon_lis[-1] + lon_step, 5))
        lat_lis[-1], lon_lis[-1] = lat1, lon1
        wide = geodesic((lat_lis[0], lon_lis[0]), (lat_lis[1], lon_lis[0])).m
        high = geodesic((lat_lis[0], lon_lis[0]), (lat_lis[0], lon_lis[1])).m
        print(f"wide: {wide}, high: {high}, area: {wide * high}")
        print(f"lat_lis: {lat_lis}", "\n")
        print(f"lon_lis: {lon_lis}", "\n")
        lat_lon = {}
        lat0 = lat_lis[0]
        for i, lat1 in enumerate(lat_lis[1:]):
            lon0 = lon_lis[0]
            for j, lon1 in enumerate(lon_lis[1:]):
                lat = round((lat0 + lat1) / 2, 5)
                lon = round((lon0 + lon1) / 2, 5)
                lat_lon[f"{i}-{j}"] = (lat, lon)
                lon0 = lon1
            lat0 = lat1

        self.lat_lis, self.lon_lis, self.lat_lon = lat_lis, lon_lis, lat_lon
        self.space = int(math.sqrt(wide ** 2 + high ** 2) / 10)
        self.time = int(self.space / 10)
        print(f"space threshold: {self.space}m, time fluctuation: {self.time}s")
        print(f"multi trajectory...")

        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=32, progress_bar=False)
        df["multi"] = df.parallel_apply(lambda row: self._multi(row.lat, row.lon, row.time), axis=1)
        df['m_time'] = df["multi"].map(lambda x: x['m_time'])
        df['m_lat'] = df["multi"].map(lambda x: x['m_lat'])
        df['m_lon'] = df["multi"].map(lambda x: x['m_lon'])
        df['m_did'] = df["multi"].map(lambda x: x['m_did'])
        df.drop(columns=['multi'], inplace=True)
        print(df.info())

        # 确保每个tid都有多源轨迹
        print(f"trajectory points num: {df.shape}")
        tid = df.tid.unique().tolist()
        mutil_tid = df[df['m_time'] != 0].tid.unique().tolist()
        no_tid = list(set(tid) - set(mutil_tid))
        self.no_tid = no_tid
        print(f"generate trajectory points num: {df[df['m_time'] != 0].shape}")
        print(f"no generate trajectory points num: {df[df['m_time'] == 0].shape}")
        print(f"no generate trajectory tid num:{len(no_tid)}")
        if self.no_tid:
            self.no_tid = {k: 0 for k in self.no_tid}
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=32, progress_bar=False)
            df['m_time'] = df.parallel_apply(lambda row: self._random(row.tid, row.time, row.m_time), axis=1)
            print(f"generate trajectory points num: {df[df['m_time'] != 0].shape}")
            print(f"no generate trajectory points num: {df[df['m_time'] == 0].shape}")
        df['did'] = 'GPS'
        df.to_csv("./multi.csv", index=False)
        print("saved successfully")

    def test_data(self):
        df = pd.read_csv("./multi.csv", dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})

        tid = random.sample(df.tid.unique().tolist(), 6000)
        tid1 = tid[:1000]
        print(tid1.__len__())
        tid2 = tid[1000: 4000]
        print(tid2.__len__())
        test1k = df.query(f"tid in {tid1}")
        test1k.to_csv("test1K.csv", index=False)
        test3k = df.query(f"tid in {tid2}")
        test3k.to_csv("test3K.csv", index=False)

        test1K = pd.read_csv("./test1K.csv", dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
        print(test1K.shape)
        test1K.drop_duplicates(inplace=True)
        print(test1K.shape)
        print('-'*50)

        test3K = pd.read_csv("./test3K.csv", dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
        print(test3K.shape)
        test3K.drop_duplicates(inplace=True)
        print(test3K.shape)
        print('-' * 50)

    def split(self):
        data = pd.read_csv("./multi.csv", dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
        data = data[['tid', 'time', 'lat', 'lon', 'did', 'm_time', 'm_lat', 'm_lon', 'm_did']]
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        data1 = data[['tid', 'time', 'lat', 'lon', 'did']].copy()
        data1.drop_duplicates(inplace=True)
        arr1 = data1.to_numpy()
        np.save("data1.npy", arr1)

        data2 = data[['tid', 'm_time', 'm_lat', 'm_lon', 'm_did']].copy()
        data2.rename(columns={'m_time': 'time', 'm_lat': 'lat', 'm_lon': 'lon', 'm_did': 'did'}, inplace=True)
        data2 = data2[data2['time'] != 0]
        data2.drop_duplicates(inplace=True)
        arr2 = data2.to_numpy()
        np.save("data2.npy", arr2)


if __name__ == "__main__":
    ProTra().main()
    # from math import cos, radians
    # deci = 5  # decimal 小数点后的精度
    # distance = 3.333 * 1000  # m  3.333
    # r = 6371393  # 地球半径 单位m
    #
    # lat_step = (distance / (r * cos(radians(0)))) * (180 / 3.1415926)
    # print(lat_step)
    # lon_step = (distance / r) * (180 / 3.1415926)
    # print(lon_step)
