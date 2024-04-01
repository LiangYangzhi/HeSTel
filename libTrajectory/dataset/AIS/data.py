import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from geopy.distance import geodesic


class ProTra(object):
    def __init__(self):
        self.no_tid = None
        self.lat_lis = None
        self.lon_lis = None
        self.lat_lon = None
        self.space = None
        self.time = None

    def __int__(self):
        pass

    def cols(self):
        df = pd.DataFrame(data=[f"{i}" for i in range(1, 10)], columns=['month'])

        def deal(p):
            print(p)
            df = pd.read_csv(f"{p}", usecols=[0, 1, 2, 3, 10, 16])
            if df.MMSI.dtype == "object":
                df['MMSI'] = pd.to_numeric(df['MMSI'], downcast="integer", errors='coerce')
                df = df.dropna(subset=['MMSI'])
                df["MMSI"] = df['MMSI'].astype(int)
                print(df.info())
            df['MMSI'] = df["MMSI"].map(lambda u: f"{int(u)}")
            df['time'] = df["BaseDateTime"].map(
                lambda t_str: datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%S').timestamp())
            df = df[["MMSI", "time", "LAT", "LON", "VesselType", "TransceiverClass"]]
            df.rename(columns={"MMSI": 'uid', "time": 'time', "LAT": 'lat', "LON": 'lon',
                               "TransceiverClass": 'did', "VesselType": "vesselType"}, inplace=True)
            df = df[['uid', 'time', 'lat', 'lon', 'did', "vesselType"]]
            df.dropna(inplace=True)
            path = f"{p[:2]}{p[6:]}"
            print(path)
            df.to_csv(f"{path}", index=False)

        def path(m):
            lis = []
            files = os.listdir(f'./{m}')
            for f in files:
                if ".csv" not in f:
                    continue
                if 'AIS' not in f:
                    continue
                # if f[4:] in f:
                #     continue
                lis.append(f'{m}/{f}')
            return lis

        df['path'] = df.apply(lambda row: path(row.month), axis=1)
        df = df.explode('path')
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=48)
        df.parallel_apply(lambda row: deal(row.data_path), axis=1)

    def vessel(self):
        data = []
        for i in range(1, 10):
            print(i)
            files = os.listdir(f'./{i}')
            for f in files:
                if ".csv" not in f:
                    continue
                if 'AIS' in f:
                    continue
                print(f)
                df = pd.read_csv(f"{i}/{f}", usecols=[0, 4, 5])
                df.drop_duplicates(inplace=True)
                data.append(df)
        data = pd.concat(data)
        data.drop_duplicates(inplace=True)
        print(data.columns)
        pd.set_option('display.max_rows', None)

        dfa = data.query("did == 'A'")
        print("A")
        print(dfa.vesselType.value_counts())
        dfb = data.query("did == 'B'")
        print("B")
        print(dfb.vesselType.value_counts())

    def tra_density(self):
        for m in range(1, 10):
            print(f"month: {m}")
            data = []
            for i in range(1, 10):
                print(i)
                df = pd.read_csv(f"{m}/2023_0{m}_0{i}.csv")
                df["uid"] = df["uid"].astype('str')
                data.append(df)
            data = pd.concat(data)

            for d in ["A", "B"]:
                dfs = data.query(f"did == '{d}'")
                dfs = dfs.groupby('uid')
                all_dis = []
                user_num = 0
                for name, df in dfs:
                    user_num += 1
                    df.sort_values(['time'], inplace=True)
                    df['latlon'] = df.apply(lambda row: (row.lat, row.lon), axis=1)
                    latlon = df.latlon.tolist()
                    length = len(latlon)
                    distance = []
                    point = 0
                    j = 0
                    for i in range(1, length):
                        dis = geodesic(latlon[j], latlon[i]).m
                        if dis > 2000:
                            distance.append(dis)
                            point += 1
                            j = i
                    if point:
                        all_dis.append(sum(distance) / point)  # 空间去噪后，每个轨迹的平均移动距离
                    if user_num >= 5000:
                        break
                print(f"{d}: {sum(all_dis) / user_num}")

    def user_tra(self):
        for i in range(1, 10):
            files = os.listdir(f'./{i}')
            data = []
            for f in files:
                if ".csv" not in f:
                    continue
                if 'AIS' in f:
                    continue
                print(f)
                df = pd.read_csv(f"{i}/{f}")
                df["uid"] = df["uid"].astype('str')
                data.append(df)

            data = pd.concat(data)
            for d in ['A', 'B']:
                dfs = data.query(f"did == '{d}'")
                dfs = dfs.groupby("uid")
                for name, df in dfs:
                    df.to_csv(f"./user_traj{i}{d}/{name}.csv", index=False)

        for d in ['A', 'B']:
            files = []
            for i in range(1, 10):
                file = os.listdir(f"./user_traj{i}{d}")
                files.append(file)

            user = sum(files, [])
            user = list(set(user))
            print(f"user num: {len(user)}")
            user = pd.DataFrame(data=user, columns=['uid'])

            def concat(f):
                print(f)
                dfs = []
                for i in range(9):
                    if f in files[i]:
                        df = pd.read_csv(f"./user_traj{i + 1}{d}/{f}")
                        df["uid"] = df["uid"].astype("str")
                        dfs.append(df)
                dfs = pd.concat(dfs)
                dfs.to_csv(f"./user_traj{d}/{f}", index=False)

            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=48)
            user.parallel_apply(lambda row: concat(row.uid), axis=1)

    def gen_tra_exp(self):
        # generate trajectory
        for d in ['A', 'B']:
            files = os.listdir(f'./user_traj{d}')
            files = pd.DataFrame(data=files, columns=['f'])

            sta = {3: [], 5: [], 7: [], 10: []}

            def split(f):
                print(f)
                df = pd.read_csv(f'./user_traj{d}/{f}')
                df['uid'] = df['uid'].astype("str")
                df.sort_values(['time'], inplace=True)
                df['diff'] = df['time'].diff()
                df['diff'].fillna(0, inplace=True)
                df.reset_index(drop=True, inplace=True)
                threshold = {3: 0, 5: 0, 7: 0, 10: 0}
                for thre in threshold:
                    split_index = df.query(f"diff >= {60 * 60 * 24 * thre}").index.tolist()

                    if len(split_index) == 0:
                        threshold[thre] = 0
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
                        df_list.append(sub_df)
                    threshold[thre] = len(df_list)
                    continue
                return threshold

            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=48)
            files['dic'] = files.parallel_apply(lambda row: split(row.f), axis=1)
            lis = files.dic.tolist()
            for dic in lis:
                for k, v in dic.items():
                    sta[k].append(v)
            for k, v in sta.items():
                sta[k] = round(sum(v) / len(v), 2)
            print(d, sta)

    def gen_tra(self):
        # generate trajectory
        inter = 5
        print(f"time: {inter}")
        for d in ['A', 'B']:
            files = os.listdir(f'./user_traj{d}')
            files = pd.DataFrame(data=files, columns=['f'])

            def split(f):
                df = pd.read_csv(f'./user_traj{d}/{f}')
                df['uid'] = df['uid'].astype("str")
                df.sort_values(['time'], inplace=True)
                df['diff'] = df['time'].diff()
                df['diff'].fillna(0, inplace=True)
                df.reset_index(drop=True, inplace=True)

                split_index = df.query(f"diff >= {60 * 60 * 24 * inter}").index.tolist()

                if len(split_index) == 0:
                    df.drop(columns=['diff'], inplace=True)
                    df['tid'] = df["uid"]
                    df.to_csv(f'./user_traj{d}/{f}', index=False)
                    print(f, 0)
                    return 0

                df_list = []
                start_i = 0
                end_i = 0
                split_index.append(df.index[-1])
                for site, i in enumerate(split_index):
                    end_i = i
                    sub_df = df.loc[start_i: end_i]
                    start_i = i + 1
                    sub_df["tid"] = sub_df["uid"].map(lambda u: f"{u}_{site + 1}")
                    df_list.append(sub_df)
                print(f, len(df_list))
                tracks = pd.concat(df_list)
                tracks.drop(columns=['diff'], inplace=True)
                tracks.to_csv(f'./user_traj{d}/{f}', index=False)
                return 1

            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=48)
            files.parallel_apply(lambda row: split(row.f), axis=1)

    def denoising(self):
        thre = 2000  # m
        print(f"space denoising threshold: {thre}m")
        for d in ['A', 'B']:
            files = os.listdir(f'./user_traj{d}')
            files = pd.DataFrame(data=files, columns=['f'])

            def space(f):
                data = pd.read_csv(f'./user_traj{d}/{f}', dtype={'uid': str, 'tid': str, 'did': str})
                print(f"{f}, number: {data.shape[0]}")
                tid_list = data.tid.unique().tolist()
                result = []
                for tid in tid_list:
                    df = data.query(f"tid == '{tid}'")
                    if df.shape[0] <= 1:
                        df['latlon'] = df.apply(lambda row: (row.lat, row.lon), axis=1)
                        df['flag'] = 1
                        result.append(df)
                        continue

                    df.sort_values(['time'], inplace=True)
                    df['latlon'] = df.apply(lambda row: (row.lat, row.lon), axis=1)
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
                result.to_csv(f"./data{d}/{f}", index=False)
                return 1

            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=42)
            files.parallel_apply(lambda row: space(row.f), axis=1)

    def density(self):
        for d in ['A', 'B']:
            files = os.listdir(f'./data{d}')
            files = pd.DataFrame(data=files, columns=['f'])

            def space(f):
                data = pd.read_csv(f'./data{d}/{f}', dtype={'uid': str, 'tid': str, 'did': str})
                tid_list = data.tid.unique().tolist()
                distance = []
                for tid in tid_list:
                    df = data.query(f"tid == '{tid}'")
                    if df.shape[0] <= 1:
                        continue

                    df.sort_values(['time'], inplace=True)
                    df['latlon'] = df.apply(lambda row: (row.lat, row.lon), axis=1)
                    latlon = df.latlon.tolist()
                    length = len(latlon) - 1
                    j = 0
                    average = []
                    for i in range(1, length):
                        dis = geodesic(latlon[j], latlon[i]).m
                        average.append(dis)
                    if average:
                        average = round(sum(average) / len(average), 2)
                        distance.append(average)
                return distance

            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=48)
            files['distance'] = files.parallel_apply(lambda row: space(row.f), axis=1)

            distance = files.distance.to_list()
            distance = sum(distance, [])
            print(f"{d} average space distance: {round(sum(distance) / len(distance), 1)}")
            print(f"{d} median space distance: {np.median(np.array(distance))}")

    def multi_tra(self):
        files = os.listdir(f'./dataA')
        files = pd.DataFrame(data=files, columns=['f'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=48)
        files['tra'] = files['f'].parallel_map(lambda f: pd.read_csv(f'./dataA/{f}', dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str}))
        df = pd.concat(files.tra.to_list())
        df.reset_index(drop=True, inplace=True)
        print(df.info())

        lat_size, lon_size = 100, 300
        print(f"lat_size: {lat_size}, lon_size: {lon_size}")
        # lat0, lat1, lon0, lon1 = -24.86131, 85.72041, -179.99747, 179.13914
        lat0, lat1 = df.lat.min(), df.lat.max()
        lon0, lon1 = df.lon.min(), df.lon.max()
        print(f"lat0: {lat0}, lat1: {lat1}, lon0: {lon0}, lon1: {lon1}")
        lat_len, lon_len = abs(lat0) + abs(lat1), abs(lon0) + abs(lon1)
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
        self.space = int(math.sqrt(wide**2 + high**2) / 10)
        self.time = int(self.space / 10)
        print(f"space threshold: {self.space}m, time fluctuation: {self.time}s")
        print(f"multi trajectory...")

        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=48)
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
            pandarallel.initialize(nb_workers=48)
            df['m_time'] = df.parallel_apply(lambda row: self._random(row.tid, row.time, row.m_time), axis=1)
            print(f"generate trajectory points num: {df[df['m_time'] != 0].shape}")
            print(f"no generate trajectory points num: {df[df['m_time'] == 0].shape}")
        df.to_csv("./multiA.csv", index=False)
        print("saved successfully")

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
            if multi_t < 1672502400:
                multi_t = 1672502400
            if multi_t > 1696089599:
                multi_t = 1696089599
            return {"m_time": multi_t, "m_lat": center[0], "m_lon": center[1], "m_did": f"{lati}-{lonj}"}
        return {"m_time": 0, "m_lat": center[0], "m_lon": center[1], "m_did": f"{lati}-{lonj}"}

    def test_data(self):
        df = pd.read_csv("./multiA.csv", dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})

        tid = random.sample(df.tid.unique().tolist(), 6000)
        # tid1 = tid[:1000]
        # print(tid1.__len__())
        # tid2 = tid[1000: 4000]
        # print(tid2.__len__())
        # test1k = df.query(f"tid in {tid1}")
        # test1k.to_csv("test1K.csv", index=False)
        # test3k = df.query(f"tid in {tid2}")
        # test3k.to_csv("test3K.csv", index=False)
        # sample6K = df.query(f"tid in {tid}")
        # sample6K.to_csv("sample6K.csv", index=False)

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

        tid1 = test1K.tid.unique().tolist()
        tid3 = test3K.tid.unique().tolist()
        tid = list(set(tid) - set(tid1) - set(tid3))
        tid = tid[: 2000]

        sample = df.query(f"tid in {tid}")
        sample6K = pd.concat([sample, test1K, test3K])
        print(sample6K.shape)
        sample6K.drop_duplicates(inplace=True)
        print(sample6K.shape)
        sample6K.to_csv("sample6K.csv", index=False)

    def split(self):
        data = pd.read_csv("./multiA.csv", dtype={
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

    def run(self):
        # self.cols()
        # self.vessel()
        # self.user_tra()
        # self.gen_tra_exp()
        # self.gen_tra()
        # self.denoising()
        # self.density()
        # self.multi_tra()
        # self.test_data()
        self.split()


if __name__ == "__main__":
    pro_tra = ProTra()
    pro_tra.run()
