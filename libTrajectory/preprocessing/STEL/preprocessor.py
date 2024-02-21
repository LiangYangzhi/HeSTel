import logging
import io
import pandas as pd
from geopy.distance import geodesic

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=48)

data_path = "./libTrajectory/dataset/AIS/"
log_path = "./libTrajectory/logs/STEL/"


class Scoor(object):
    def __init__(self, data, tseg):
        self.data = data
        self.tseg = tseg
        self.sthre = 1000  # 空间坐标系粒度中空间点个数阈值

        logging.info(f"时间分割编号={tseg}, data shape={self.data.shape}, self.sthre = {self.sthre}")
        self.run()

    def run(self):
        self.data.loc[:, 'scoor'] = ''  # 初始化
        self.coor = {'': {"lat0": self.data.lat.min(), "lat1": self.data.lat.max(),
                          "lon0": self.data.lon.min(), "lon1": self.data.lon.max()}}
        self.scoor = []  # 存放最终空间坐标系
        split_cell = ['']
        while True:
            for cell in split_cell:
                latlon = self.coor[cell]
                self._ssub(cell, latlon)  # split sub cell
            # update the cell name to which the point belongs
            updf = self.data[self.data.scoor.isin(list(split_cell))].copy()  # 需要更新的数据
            noupdf = self.data[~self.data.scoor.isin(list(split_cell))].copy()
            updf['scoor'] = updf.apply(lambda row: self._scell(row.lat, row.lon, row.scoor), axis=1)
            self.data = pd.concat([updf, noupdf])
            # judge the points number of cell
            split_cell = {cell: num for cell, num in self.data.scoor.value_counts().to_dict().items() if num > self.sthre}
            if split_cell.__len__() == 0:
                break

        for s, dic in self.coor.items():
            self.scoor.append([self.tseg, s, dic['lat0'], dic['lat1'], dic['lon0'], dic['lon1']])

    def _ssub(self, cell, latlon):
        lat = (latlon['lat0'] + latlon['lat1']) / 2
        lon = (latlon['lon0'] + latlon['lon1']) / 2
        self.coor[f'{cell}0'] = {"lat0": lat, "lat1": latlon['lat1'], "lon0": latlon['lon0'], "lon1": lon}
        self.coor[f'{cell}1'] = {"lat0": lat, "lat1": latlon['lat1'], "lon0": lon, "lon1": latlon['lon1']}
        self.coor[f'{cell}2'] = {"lat0": latlon['lat0'], "lat1": lat, "lon0": lon, "lon1": latlon['lon1']}
        self.coor[f'{cell}3'] = {"lat0": latlon['lat0'], "lat1": lat, "lon0": latlon['lon0'], "lon1": lon}

    def _scell(self, lat, lon, cell):
        coor = 0
        for i in range(4):
            coor = self.coor[f"{cell}{i}"]
            if coor["lat0"] <= lat <= coor["lat1"] and coor["lon0"] <= lon <= coor["lon1"]:
                return f"{cell}{i}"
        raise logging.critical(f"current cell is {cell}, cell_latlon: {coor}, point={(lat, lon)} no find 4split cell")

    def get(self):
        return [self.data, self.scoor]


class Tcoor(object):
    def __init__(self, data, sseg):
        self.data = data
        self.sseg = sseg
        self.tthre = 1000  # 时间坐标系粒度中时间点个数阈值

        logging.info(f"空间分割编号={sseg}, data shape={self.data.shape}, self.tthre = {self.tthre}")
        self.run()

    def run(self):
        self.data.loc[:, 'tcoor'] = ''  # 初始化
        self.coor = {'': {"t0": self.data.time.min(), "t1": self.data.time.max()}}
        self.tcoor = []
        split_cell = ['']
        while True:
            for cell in split_cell:
                t01 = self.coor[cell]
                self._tsub(cell, t01)
            #  update the cell name to which the point belongs
            updf = self.data[self.data.tcoor.isin(list(split_cell))].copy()  # 需要更新的数据
            noupdf = self.data[~self.data.tcoor.isin(list(split_cell))].copy()
            updf['tcoor'] = updf.apply(lambda row: self._tcell(row.time, row.tcoor), axis=1)
            self.data = pd.concat([updf, noupdf])
            # "judge the points number of cell"
            split_cell = {cell: num for cell, num in self.data.tcoor.value_counts().to_dict().items() if num > self.tthre}
            if split_cell.__len__() == 0:
                break

        for t, dic in self.coor.items():
            self.tcoor.append([self.sseg, t, dic['t0'], dic['t1']])

    def _tsub(self, cell, t01):
        sub = 4  # 切分粒度
        length = t01['t1'] - t01['t0']
        step = int(length / sub)
        nodes = [t01['t0']]
        for _ in range(sub):
            nodes.append(nodes[-1] + step)
        nodes[-1] = t01['t1']
        for i in range(sub):
            self.coor[f"{cell}{i}"] = {"t0": nodes[i], "t1": nodes[i + 1]}

    def _tcell(self, t, cell):
        t01 = 0
        for i in range(4):
            t01 = self.coor[f"{cell}{i}"]
            if t01['t0'] <= t <= t01['t1']:
                return f"{cell}{i}"
        raise logging.info(f"current cell is {cell}, cell_t01: {t01}, current point t={t} no find 4split cell")

    def get(self):
        return [self.data, self.tcoor]


class Preprocessor(object):
    def __init__(self, data_path, test_path={}):
        self.data_path = data_path
        self.test_path = test_path
        self.data1 = None  # Active
        self.data2 = None  # Passive
        self.inter = 60 * 60 * 24  # 时间分割下的时间间隔阈值
        self.dis = 100 * 1000  # 空间划分下的空间距离阈值 单位m

        logging.info(f"self.path = {self.data_path}, self.inter = {self.inter}, self.dis = {self.dis}")

    def run(self):
        self.loader()
        self.cleaner()
        # Active
        self.tseg()  # time segment
        self.scoor()  # space coordinate
        self.ts2vector()  # to vector
        # Passive
        self.sseg()  # space segment
        self.tcoor()  # time coordinate
        self.st2vector()  # to vector

    def loader(self):
        logging.info("data loading...")
        data = pd.read_csv(self.data_path, dtype={
            'uid': str, 'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str,
            'm_time': int, 'm_lat': float, 'm_lon': float, 'm_did': str})
        self.data1 = data[['tid', 'time', 'lat', 'lon', 'did']].copy()
        self.data2 = data[['tid', 'm_time', 'm_lat', 'm_lon', 'm_did']].copy()
        self.data2.rename(columns={'m_time': 'time', 'm_lat': 'lat', 'm_lon': 'lon', 'm_did': 'did'}, inplace=True)
        self.data2 = self.data2[self.data2['time'] != 0]

        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info: {buffer.getvalue()}")
        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info: {buffer.getvalue()}")

        self.test = {}
        for k, v in self.test_path.items():
            data = pd.read_csv(v, usecols=['tid'], dtype={'tid': str})
            self.test[k] = data

        logging.info("data load completed")

    def cleaner(self):
        logging.info("data clean...")
        self.data1.dropna(inplace=True)
        self.data1.drop_duplicates(inplace=True)
        self.data1.sort_values(['time'], inplace=True)
        self.data2.dropna(inplace=True)
        self.data2.drop_duplicates(inplace=True)
        self.data2.sort_values(['time'], inplace=True)
        logging.info("data clean completed")

    def tseg(self):
        logging.info("time segment...")
        t0 = self.data1.time.min()
        t1 = self.data1.time.max()
        t_len = t1 - t0
        t_size = int(t_len / self.inter)
        t_step = int(t_len / t_size)
        logging.info(f"start time: {t0}, end time: {t1}, time length: {t_len}, time interval: {self.inter}, num of time segment: {t_size}")
        t_lis = [t0]
        for _ in range(t_size):
            t_lis.append(int(t_lis[-1] + t_step))
        t_lis[-1] = t1
        self.t_lis = t_lis

        self.data1['tseg'] = self.data1['time'].map(lambda t: self._tsegf(t))
        logging.info(f"data1 tseg unique length: {self.data1.tseg.unique().__len__()}")

        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info after time segment: {buffer.getvalue()}")
        logging.info("time segment completed")

    def _tsegf(self, t):
        # time segment flag
        t0 = self.t_lis[0]
        for i, t1 in enumerate(self.t_lis[1:]):
            if t0 <= t <= t1:
                return i
            t0 = t1
        logging.critical(f"t={t} 找不到相应的时间分割编号")

    def scoor(self):
        logging.info("space coordinate...")
        tseg = self.data1[['tseg']].copy()
        tseg.drop_duplicates(inplace=True)
        tseg['df'] = tseg['tseg'].map(lambda t: self.data1.query(f"tseg == {t}").copy())
        tseg['df_scoor'] = tseg.parallel_apply(lambda row: Scoor(row.df, row.tseg).get(), axis=1)  # parallel_apply
        tseg['df'] = tseg['df_scoor'].map(lambda i: i[0])
        tseg['scoor'] = tseg['df_scoor'].map(lambda i: i[1])

        scoor = sum(tseg.scoor.tolist(), [])
        scoor = pd.DataFrame(data=scoor, columns=['tseg', 'scoor', 'lat0', 'lat1', 'lon0', 'lon1'])
        scoor.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_scoor.csv", index=False)

        self.data1 = pd.concat(tseg.df.tolist())
        self.data1.sort_values(['time'], inplace=True)
        self.data1.reset_index(inplace=True, drop=True)
        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info after space coordinate: {buffer.getvalue()}")
        logging.info("space coordinate completed")

    def sseg(self):
        logging.info("space segment...")
        deci = 5  # decimal 小数点后的精度
        lat0, lat1 = self.data2.lat.min(), self.data2.lat.max()
        lon0, lon1 = self.data2.lon.min(), self.data2.lon.max()
        logging.info(f"lat0: {lat0}, lat1: {lat1}, lon0: {lon0}, lon1: {lon1}")

        # 通过lat、lon距离长度 和 距离阈值 获取 lat、lon分段的数量
        latdis, londis = geodesic((lat0, 0), (lat1, 0)).m, geodesic((0, lon0), (0, lon1)).m
        lat_size, lon_size = int(latdis / self.dis), int(londis / self.dis)
        logging.info(f"lat_size: {lat_size}, lon_size： {lon_size}， distance threshold: {self.dis}")

        # 获取区域的lat、lon步长
        lat_len, lon_len = abs(lat0) + abs(lat1), abs(lon0) + abs(lon1)
        lat_step, lon_step = round(lat_len / lat_size, deci), round(lon_len / lon_size, deci)
        logging.info(f"lat_len: {lat_len}, lon_len： {lon_len}， lat_step: {lat_step}, lon_step: {lon_step}")

        # 生成区域节点的lat和lon值
        lat_lis, lon_lis = [lat0], [lon0]
        for _ in range(lat_size):
            lat_lis.append(round(lat_lis[-1] + lat_step, deci))
        for _ in range(lon_size):
            lon_lis.append(round(lon_lis[-1] + lon_step, deci))
        lat_lis[-1], lon_lis[-1] = lat1, lon1

        self.lat_lis, self.lon_lis = lat_lis, lon_lis
        self.data2['sseg'] = self.data2.apply(lambda row: self._ssegf(row.lat, row.lon), axis=1)
        logging.info(f"data2 sseg unique length: {self.data2.sseg.unique().__len__()}")
        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info after time segment: {buffer.getvalue()}")
        logging.info("space segment completed")

    def _ssegf(self, lat, lon):
        # space segment flag
        latf = None
        lat0 = self.lat_lis[0]
        for i, lat1 in enumerate(self.lat_lis[1:]):
            if lat0 <= lat <= lat1:
                latf = i
                break
            lat0 = lat1
        if latf is None:
            logging.critical(f"lat={lat} 找不到对应的空间切割编号")

        lonf = None
        lon0 = self.lon_lis[0]
        for j, lon1 in enumerate(self.lon_lis[1:]):
            if lon0 <= lon <= lon1:
                lonf = j
                break
            lon0 = lon1
        if lonf is None:
            logging.critical(f"lon={lon} 找不到对应的空间切割编号")

        return f"{latf}-{lonf}"

    def tcoor(self):
        logging.info("time coordinate...")
        sseg = self.data2[['sseg']].copy()
        sseg.drop_duplicates(inplace=True)
        sseg['df'] = sseg['sseg'].map(lambda s: self.data2.query(f"sseg == '{s}'").copy())
        sseg['df_tcoor'] = sseg.parallel_apply(lambda row: Tcoor(row.df, row.sseg).get(), axis=1)  # parallel_apply
        sseg['df'] = sseg['df_tcoor'].map(lambda i: i[0])
        sseg['tcoor'] = sseg['df_tcoor'].map(lambda i: i[1])

        tcoor = sum(sseg.tcoor.tolist(), [])
        tcoor = pd.DataFrame(data=tcoor, columns=['sseg', 'tcoor', 't0', 't1'])
        tcoor.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_tcoor.csv", index=False)

        self.data2 = pd.concat(sseg.df.tolist())
        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info after time coordinate: {buffer.getvalue()}")
        logging.info("time coordinate system completed")

    def ts2vector(self):
        logging.info("time segment and space coordinate --> vector...")
        self.data1["tsid"] = self.data1.apply(lambda row: f"{row.tseg}_{row.scoor}", axis=1)
        self.ts_vec = self.data1[['tseg', 'scoor']].copy()
        self.ts_vec.drop_duplicates(inplace=True)

        tv0 = [0] * (self.ts_vec.tseg.max() + 1)  # 初始vector，+1:tseg从0开始
        logging.info(f"time vector length: {len(tv0)}")
        self.ts_vec["tv"] = self.ts_vec['tseg'].map(lambda tseg: self._tseg2v(tseg, tv0))

        self.ts_vec["sv"] = self.ts_vec['scoor'].map(lambda scoor: ''.join(map(self._coor2v, str(scoor))))
        svlen = max([len(str(i)) for i in self.ts_vec.scoor.unique().tolist()]) * 4  # sv max length
        logging.info(f"space vector length: {svlen}")
        self.ts_vec["sv"] = self.ts_vec["sv"].map(lambda sv: sv.ljust(svlen, '0'))
        self.ts_vec["sv"] = self.ts_vec["sv"].map(lambda sv: [int(i) for i in sv])

        logging.info(f"time space vector length: {len(tv0) + svlen}")
        self.ts_vec["vector"] = self.ts_vec.apply(lambda row: row.tv + row.sv, axis=1)
        self.ts_vec["tsid"] = self.ts_vec.apply(lambda row: f"{row.tseg}_{row.scoor}", axis=1)

        buffer = io.StringIO()
        self.ts_vec.info(buf=buffer)
        logging.info(f"ts_vec info: {buffer.getvalue()}")
        logging.info("vector completed")

    def _tseg2v(self, tseg, tv0):
        tv0[tseg] = 1
        return tv0

    def _coor2v(self, scoor):
        binary_dict = {
            '0': '0001',
            '1': '0010',
            '2': '0100',
            '3': '1000'
        }
        return binary_dict[scoor]

    def st2vector(self):
        logging.info("space segment and time coordinate --> vector...")
        self.data2["stid"] = self.data2.apply(lambda row: f"{row.sseg}_{row.tcoor}", axis=1)
        self.st_vec = self.data2[['sseg', 'tcoor']].copy()
        self.st_vec.drop_duplicates(inplace=True)

        sseg = self.st_vec.sseg.unique().tolist()
        # 初始vector
        latv0 = [0] * (max([int(s.split("-")[0]) for s in sseg]) + 1)  # 从0开始标记(max + 1)
        lonv0 = [0] * (max([int(s.split("-")[1]) for s in sseg]) + 1)  # 从0开始标记
        logging.info(f"lat vector length: {len(latv0)}, lon vector length: {len(lonv0)}, space vector length: {len(latv0) + len(lonv0)}")
        self.st_vec["sv"] = self.st_vec['sseg'].map(lambda sseg: self._sseg2v(sseg, latv0, lonv0))

        self.st_vec['tv'] = self.st_vec['tcoor'].map(lambda tcoor: ''.join(map(self._coor2v, str(tcoor))))
        tvlen = max([len(str(i)) for i in self.st_vec.tcoor.unique().tolist()]) * 4  # tv max length
        logging.info(f"time vector length: {tvlen}")
        self.st_vec["tv"] = self.st_vec["tv"].map(lambda tv: tv.ljust(tvlen, '0'))
        self.st_vec["tv"] = self.st_vec["tv"].map(lambda tv: [int(i) for i in tv])

        logging.info(f"space time vector length: {len(latv0) + len(lonv0) + tvlen}")
        self.st_vec["vector"] = self.st_vec.apply(lambda row: row.sv + row.tv, axis=1)
        self.st_vec["stid"] = self.st_vec.apply(lambda row: f"{row.sseg}_{row.tcoor}", axis=1)

        buffer = io.StringIO()
        self.st_vec.info(buf=buffer)
        logging.info(f"st_vec info: {buffer.getvalue()}")
        logging.info("vector completed")

    def _sseg2v(self, sseg, latv0, lonv0):
        latf = int(sseg.split("-")[0])
        lonf = int(sseg.split("-")[1])
        latv0[latf] = 1
        lonv0[lonf] = 1
        v = latv0 + lonv0
        return v

    def get(self):
        if self.data1 is None:
            dic1 = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str, 'tseg': int, 'scoor': str}
            dic2 = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str, 'sseg': str, 'tcoor': str}
            train_data1 = pd.read_csv(f"{log_path}/{self.data_path.split('/')[-1]}_train_data1.csv", dtype=dic1)
            train_data2 = pd.read_csv(f"{log_path}/{self.data_path.split('/')[-1]}_train_data2.csv", dtype=dic2)
            train = [train_data1, train_data2]
            test_data = {}
            for k, v in self.test_path.items():
                data1 = pd.read_csv(f"{log_path}/{v.split('/')[-1]}_{k}_data1.csv", dtype=dic1)
                data2 = pd.read_csv(f"{log_path}/{v.split('/')[-1]}_{k}_data2.csv", dtype=dic2)
                test_data[k] = [data1, data2]

            dicts = {'tseg': int, 'scoor': str, 'tv': str, 'sv': str, 'vector': str, 'tsid': str}
            ts_vec = pd.read_csv(f"{log_path}/{self.data_path.split('/')[-1]}_ts_vec.csv", dtype=dicts)
            ts_vec['vector'] = ts_vec['vector'].map(lambda v: eval(v))
            dicst = {'sseg': str, 'tcoor': str, 'sv': str, 'tv': str, 'vector': str, 'stid': str}
            st_vec = pd.read_csv(f"{log_path}/{self.data_path.split('/')[-1]}_st_vec.csv", dtype=dicst)
            st_vec['vector'] = st_vec['vector'].map(lambda v: eval(v))
            self.ts_vec, self.st_vec = ts_vec, st_vec

        else:
            # active、passive
            test_tid = []
            test_data = {}
            for k, df in self.test.items():
                tid = df.tid.unique().tolist()
                v = self.test_path[k]
                test_tid += tid
                data1 = self.data1.query(f"tid in {tid}").copy()
                data1.reset_index(drop=True, inplace=True)
                data1.to_csv(f"{log_path}/{v.split('/')[-1]}_{k}_data1.csv", index=False)
                data2 = self.data2.query(f"tid in {tid}").copy()
                data2.reset_index(drop=True, inplace=True)
                data2.to_csv(f"{log_path}/{v.split('/')[-1]}_{k}_data2.csv", index=False)
                test_data[k] = [data1, data2]

            data1 = self.data1.query(f"tid not in {test_tid}").copy()
            data1.reset_index(drop=True, inplace=True)
            data1.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_train_data1.csv", index=False)
            data2 = self.data2.query(f"tid not in {test_tid}").copy()
            data2.reset_index(drop=True, inplace=True)
            data2.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_train_data2.csv", index=False)
            train = [data1, data2]

            self.ts_vec.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_ts_vec.csv", index=False)
            self.st_vec.to_csv(f"{log_path}/{self.data_path.split('/')[-1]}_st_vec.csv", index=False)

        tsid = [train[0][['tsid']]]
        for lis in test_data.values():
            tsid.append(lis[0][['tsid']])
        tsid = pd.concat(tsid)
        tsid_counts = tsid.tsid.value_counts().to_dict()

        stid = [train[1][['stid']]]
        for lis in test_data.values():
            stid.append(lis[1][['stid']])
        stid = pd.concat(stid)
        stid_counts = stid.stid.value_counts().to_dict()

        return train, test_data, self.ts_vec, self.st_vec, tsid_counts, stid_counts
