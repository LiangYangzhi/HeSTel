import logging
import io
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import torch
from torch_geometric.data import Data

# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=48)

logging.basicConfig(filename='example.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info("-" * 50)


class Preprocessor(object):
    def __init__(self, path):
        self.path = path
        self.data1 = None  # Active
        self.data2 = None  # Passive

        self.inter = 60 * 60 * 24  # 时间分割下的时间间隔阈值
        self.dis = 1000 * 1000  # 空间划分下的空间距离阈值 单位m
        self.sthre = 100  # 空间坐标系粒度中空间点个数阈值
        self.tthre = 100  # 时间坐标系粒度中时间点个数阈值

        logging.basicConfig(filename='example.logs')
        logging.info(f"self.path = {self.path}, self.inter = {self.inter}, self.dis = {self.dis}, "
                     f"self.sthre = {self.sthre}, self.tthre = {self.tthre}")

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
        # create graph
        self.create_graph()

    def loader(self):
        logging.info("data loading...")
        data = pd.read_csv(self.path, dtype={
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
        logging.info("data load completed")

    def cleaner(self):
        logging.info("data clean...")
        self.data1.dropna(inplace=True)
        self.data1.drop_duplicates(inplace=True)
        self.data2.dropna(inplace=True)
        self.data2.drop_duplicates(inplace=True)
        logging.info("data clean completed")

    def tseg(self):
        logging.info("time segment...")
        t0 = self.data1.time.min()
        t1 = self.data1.time.max()
        t_len = t1 - t0
        t_size = int(t_len / self.inter)
        t_step = int(t_len / t_size)
        logging.info(f"start time: {t0}, end time: {t1}, time length: {t_len}, "
                     f"time interval: {self.inter}, num of time segment: {t_size}")
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
        tseg = self.data1.tseg.unique().tolist()

        df_lis = []
        scoor = []  # [t, s, lat0, lat1, lon0, lon1]
        for t in tseg:
            df = self.data1.query(f"tseg == {t}").copy()
            logging.info(f"时间分割编号={t}, data shape={df.shape}")
            df.loc[:, 'scoor'] = ''  # 初始化
            self.coor = {'': {"lat0": df.lat.min(), "lat1": df.lat.max(), "lon0": df.lon.min(), "lon1": df.lon.max()}}
            split_cell = ['']
            while True:
                for cell in split_cell:
                    latlon = self.coor[cell]
                    self._ssub(cell, latlon)  # split sub cell
                # update the cell name to which the point belongs
                updf = df[df.scoor.isin(list(split_cell))].copy()  # 需要更新的数据
                noupdf = df[~df.scoor.isin(list(split_cell))].copy()
                updf['scoor'] = updf.apply(lambda row: self._scell(row.lat, row.lon, row.scoor), axis=1)
                df = pd.concat([updf, noupdf])
                # judge the points number of cell
                split_cell = {cell: num for cell, num in df.scoor.value_counts().to_dict().items() if num > self.sthre}
                if split_cell.__len__() == 0:
                    break

            df_lis.append(df)
            for s, dic in self.coor.items():
                scoor.append([t, s, dic['lat0'], dic['lat1'], dic['lon0'], dic['lon1']])

        scoor = pd.DataFrame(data=scoor, columns=['tseg', 'scoor', 'lat0', 'lat1', 'lon0', 'lon1'])
        scoor.to_csv('scoor.csv', index=False)

        self.data1 = pd.concat(df_lis)
        self.data1.sort_values(['time'], inplace=True)
        self.data1.reset_index(inplace=True, drop=True)
        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info after space coordinate: {buffer.getvalue()}")
        logging.info("space coordinate completed")

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
        logging.info("time coordinate system...")
        sseg = self.data2.sseg.unique().tolist()

        df_lis = []
        tcoor = []  # [s, t, t0, t1]
        for s in sseg:
            df = self.data2.query(f"sseg == '{s}'").copy()
            logging.info(f"空间分割编号={s}, data shape={df.shape}")
            df.loc[:, 'tcoor'] = ''  # 初始化
            self.coor = {'': {"t0": df.time.min(), "t1": df.time.max()}}
            split_cell = ['']
            while True:
                for cell in split_cell:
                    t01 = self.coor[cell]
                    self._tsub(cell, t01)
                #  update the cell name to which the point belongs
                updf = df[df.tcoor.isin(list(split_cell))].copy()  # 需要更新的数据
                noupdf = df[~df.tcoor.isin(list(split_cell))].copy()
                updf['tcoor'] = updf.apply(lambda row: self._tcell(row.time, row.tcoor), axis=1)
                df = pd.concat([updf, noupdf])
                # "judge the points number of cell"
                split_cell = {cell: num for cell, num in df.tcoor.value_counts().to_dict().items() if num > self.tthre}
                if split_cell.__len__() == 0:
                    break

            df_lis.append(df)
            for t, dic in self.coor.items():
                tcoor.append([s, t, dic['t0'], dic['t1']])

        tcoor = pd.DataFrame(data=tcoor, columns=['sseg', 'tcoor', 't0', 't1'])
        tcoor.to_csv('tcoor.csv', index=False)

        self.data2 = pd.concat(df_lis)
        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info after time coordinate: {buffer.getvalue()}")
        logging.info("time coordinate system completed")

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

    def ts2vector(self):
        logging.info("time segment and space coordinate --> vector...")
        tv0 = [0] * (self.data1.tseg.max() + 1)  # 初始vector，+1:tseg从0开始
        logging.info(f"time vector length: {len(tv0)}")
        self.data1["tv"] = self.data1['tseg'].map(lambda tseg: self._tseg2v(tseg, tv0))

        self.data1["sv"] = self.data1['scoor'].map(lambda scoor: ''.join(map(self._coor2v, str(scoor))))
        svlen = max([len(str(i)) for i in self.data1.scoor.unique().tolist()]) * 4  # sv max length
        logging.info(f"space vector length: {svlen}")
        self.data1["sv"] = self.data1["sv"].map(lambda sv: sv.ljust(svlen, '0'))
        self.data1["sv"] = self.data1["sv"].map(lambda sv: [int(i) for i in sv])

        logging.info(f"time space vector length: {len(tv0) + svlen}")
        self.data1["tsvec"] = self.data1.apply(lambda row: row.tv + row.sv, axis=1)

        buffer = io.StringIO()
        self.data1.info(buf=buffer)
        logging.info(f"data1 info after time segment: {buffer.getvalue()}")
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
        sseg = self.data2.sseg.unique().tolist()
        # 初始vector
        latv0 = [0] * (max([int(s.split("-")[0]) for s in sseg]) + 1)  # 从0开始标记(max + 1)
        lonv0 = [0] * (max([int(s.split("-")[1]) for s in sseg]) + 1)  # 从0开始标记
        logging.info(f"lat vector length: {len(latv0)}, lon vector length: {len(lonv0)}, space vector length: {len(latv0) + len(lonv0)}")
        self.data2["sv"] = self.data2['sseg'].map(lambda sseg: self._sseg2v(sseg, latv0, lonv0))

        self.data2['tv'] = self.data2['tcoor'].map(lambda tcoor: ''.join(map(self._coor2v, str(tcoor))))
        tvlen = max([len(str(i)) for i in self.data2.tcoor.unique().tolist()]) * 4  # tv max length
        logging.info(f"time vector length: {tvlen}")
        self.data2["tv"] = self.data2["tv"].map(lambda tv: tv.ljust(tvlen, '0'))
        self.data2["tv"] = self.data2["tv"].map(lambda tv: [int(i) for i in tv])

        logging.info(f"space time vector length: {len(latv0) + len(lonv0) + tvlen}")
        self.data2["stvec"] = self.data2.apply(lambda row: row.sv + row.tv, axis=1)

        buffer = io.StringIO()
        self.data2.info(buf=buffer)
        logging.info(f"data2 info after time coordinate: {buffer.getvalue()}")
        logging.info("vector completed")

    def _sseg2v(self, sseg, latv0, lonv0):
        latf = int(sseg.split("-")[0])
        lonf = int(sseg.split("-")[1])
        latv0[latf] = 1
        lonv0[lonf] = 1
        v = latv0 + lonv0
        return v

    def ts2graph(self):
        logging.info("time segment and space coordinate --> graph...")
        self.graph1 = pd.DataFrame(data=self.data1.tid.unique().tolist(), columns=['tid'])
        self.graph1.info()
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=12, progress_bar=True)
        self.graph1['node_edge'] = self.graph1['tid'].parallel_map(lambda tid: self._ts2ne(tid))
        self.tid1 = self.graph1.tid.tolist()
        self.graph1 = self.graph1.node_edge.tolist()
        logging.info("time segment and space coordinate graph completed")

    def _ts2ne(self, tid):
        dft = self.data1.query(f"tid == '{tid}'").copy()
        dft.sort_values(['time'], inplace=True)
        dft.reset_index(drop=True, inplace=True)
        # nodel
        node = dft.tsvec.tolist()

        # edge index and edge attribute
        edge_ind = []
        edge_attr = []
        dfg = dft.groupby('tseg')
        for _, sub_df in dfg:
            if sub_df.shape[0] == 1:
                break
            # The spatial distance between two nodes
            sub_df['ind0'] = sub_df.apply(lambda row: row.name, axis=1)
            sub_df['latlon0'] = sub_df.apply(lambda row: (row.lat, row.lon), axis=1)
            df0 = sub_df[['ind0', 'latlon0', 'scoor']].copy()
            df1 = df0.copy()
            df1.rename(columns={'ind0': 'ind1', 'latlon0': 'latlon1'}, inplace=True)
            df = df0.merge(df1, how='outer', on='scoor')
            df.drop_duplicates(subset=['ind0', 'ind1'], inplace=True)
            if df.shape[0] == 0:
                break
            df['edge_ind'] = df.apply(lambda row: [row.ind0, row.ind1], axis=1)
            for ind in df.edge_ind.tolist():
                edge_ind.append(ind)
            df['edge_attr'] = df.apply(lambda row: 1 / (geodesic(row.latlon0, row.latlon1).m + 0.01), axis=1)
            for attr in df.edge_attr.tolist():
                edge_attr.append(attr)

        return [torch.tensor(node), torch.tensor(edge_ind), torch.tensor(edge_attr)]
        # return Data(x=torch.tensor(node), edge_index=torch.tensor(edge_ind), edge_attr=torch.tensor(edge_attr))

    def st2graph(self):
        logging.info("space segment and time coordinate --> graph")
        self.graph2 = pd.DataFrame(data=self.data2.tid.unique().tolist(), columns=['tid'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=12, progress_bar=True)
        self.graph2['node_edge'] = self.graph2['tid'].parallel_map(lambda tid: self._st2ne(tid))
        self.tid2 = self.graph2.tid.tolist()
        self.graph2 = self.graph2.node_edge.tolist()
        logging.info("time segment and space coordinate graph completed")
        
    def _st2ne(self, tid):
        dft = self.data2.query(f"tid == '{tid}'").copy()
        dft.sort_values(['time'], inplace=True)                                                                        
        dft.reset_index(drop=True, inplace=True)                                                                       
        # nodel                                                                                                        
        node = dft.stvec.tolist()
        logging.info(f"node2={len(node[0])}")

        # edge index and edge attribute
        edge_ind = []                                                                                                  
        edge_attr = []                                                                                                  
        dfg = dft.groupby('sseg')                                                                                      
        for _, sub_df in dfg:
            if sub_df.shape[0] == 1:                                                                                   
                break
            # The time interval between two nodes
            sub_df['ind0'] = sub_df.apply(lambda row: row.name, axis=1)
            sub_df['time0'] = sub_df.apply(lambda row: row.time, axis=1)
            df0 = sub_df[['ind0', 'time0', 'tcoor']].copy()
            df1 = df0.copy()                                                                                           
            df1.rename(columns={'ind0': 'ind1', 'time0': 'time1'}, inplace=True)

            df = df0.merge(df1, how='outer', on='tcoor')
            df.drop_duplicates(subset=['ind0', 'ind1'], inplace=True)
            if df.shape[0] == 0:
                break

            df['edge_ind'] = df.apply(lambda row: [row.ind0, row.ind1], axis=1)
            for ind in df.edge_ind.tolist():
                edge_ind.append(ind)
            df['edge_attr'] = df.apply(lambda row: 1 / (abs(row.time0 - row.time1) + 1), axis=1)
            for attr in df.edge_attr.tolist():
                edge_attr.append(attr)

        return [torch.tensor(node), torch.tensor(edge_ind), torch.tensor(edge_attr)]
        # return Data(x=torch.tensor(node), edge_index=torch.tensor(edge_ind), edge_attr=torch.tensor(edge_attr))

    def create_graph(self):
        logging.info("trajectory points as nodes...")
        dim1 = len(self.data1.tsvec[0])
        dim2 = len(self.data2.stvec[0])
        dim = dim1 if dim1 > dim2 else dim2
        logging.info(f"dimension1={dim1}, dimension2={dim2}")
        fill_1 = dim - dim1
        if fill_1:
            self.data1['tsvec'] = self.data1['tsvec'].map(lambda v: v + [0] * fill_1)
            logging.info(f"dimension1={len(self.data1.tsvec[0])}, fill 0 length={fill_1}")
        fill_2 = dim - dim2
        if fill_2:
            self.data2['stvec'] = self.data2['stvec'].map(lambda v: v + [0] * fill_2)
            logging.info(f"dimension2={len(self.data2.stvec[0])} , fill 0 length={fill_2}")

        logging.info("time segment and space coordinate --> graph...")
        self.graph1 = pd.DataFrame(data=self.data1.tid.unique().tolist(), columns=['tid'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        self.graph1['node_edge'] = self.graph1['tid'].parallel_map(lambda tid: self._ts2ne(tid))
        self.tid1 = self.graph1.tid.tolist()
        self.graph1 = self.graph1.node_edge.tolist()
        logging.info(f"node1 shape={self.graph1[0][0].shape}")
        logging.info("time segment and space coordinate graph completed")

        logging.info("space segment and time coordinate --> graph")
        self.graph2 = pd.DataFrame(data=self.data2.tid.unique().tolist(), columns=['tid'])
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=24, progress_bar=True)
        self.graph2['node_edge'] = self.graph2['tid'].parallel_map(lambda tid: self._st2ne(tid))
        self.tid2 = self.graph2.tid.tolist()
        self.graph2 = self.graph2.node_edge.tolist()
        logging.info(f"node2 shape={self.graph2[0][0].shape}")
        logging.info("time segment and space coordinate graph completed")


    def get(self):
        # active、passive
        return self.tid1, self.graph1, self.tid2, self.graph2


# if __name__ == "__main__":
#     preprocessor = Preprocessor("../../dataset/AIS/test10.csv")
#     preprocessor.run()

