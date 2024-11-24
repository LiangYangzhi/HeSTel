import logging
from geopy.distance import geodesic
import numpy as np
import pandas as pd


def main(path):
    logging.info(f"path: {path}")

    logging.info("data loading...")
    columns = ['tid', 'time', 'lat', 'lon', 'stid']
    data1 = np.load(f"{path}data1.npy", allow_pickle=True)
    data1 = pd.DataFrame(data1, columns=columns).infer_objects()
    data1 = data1[columns[:-1]]

    data2 = np.load(f"{path}data2.npy", allow_pickle=True)
    data2 = pd.DataFrame(data2, columns=columns).infer_objects()
    data2 = data2[columns[:-1]]

    logging.info("numbers of entity...")
    logging.info(f"data1 vessel num: {len(data1.tid.unique())}")
    logging.info(f"data2 vessel num: {len(data2.tid.unique())}")

    logging.info("time...")
    t1min, t1max = data1.time.min(), data1.time.max()
    logging.info(f"data1 time min: {t1min}, time max: {t1max}, time duration: {t1max - t1min}")
    group1 = data1.groupby("tid").agg({"time": list})
    group1['time'] = group1['time'].map(lambda row: sorted(row))
    group1['duration'] = group1['time'].map(lambda row: max(row) - min(row))
    logging.info(f"data1 trajectory time duration: {group1.duration.mean()} s")
    group1['interval'] = group1['time'].map(lambda row: np.mean(np.diff(row)))
    logging.info(f"data1 trajectory time interval: {group1.interval.mean()} s")

    t2min, t2max = data2.time.min(), data2.time.max()
    logging.info(f"data2 time min: {t2min}, time max: {t2max}, time duration: {t2max - t2min}")
    group2 = data2.groupby("tid").agg({"time": list})
    group2['time'] = group2['time'].map(lambda row: sorted(row))
    group2['duration'] = group2['time'].map(lambda row: max(row) - min(row))
    logging.info(f"data2 trajectory time duration: {group2.duration.mean()} s")
    group2['interval'] = group2['time'].map(lambda row: np.mean(np.diff(row)))
    logging.info(f"data2 trajectory time interval: {group2.interval.mean()} s")

    logging.info("length...")
    logging.info(f"data1 points num: {data1.shape[0]}")
    group1 = data1.groupby("tid").agg({"time": list})
    group1['len'] = group1['time'].map(lambda row: len(row))
    logging.info(f"data1 trajectory length: {group1.len.mean()}")

    logging.info(f"data2 points num: {data2.shape[0]}")
    group2 = data2.groupby("tid").agg({"time": list})
    group2['len'] = group2['time'].map(lambda row: len(row))
    logging.info(f"data2 trajectory length: {group2.len.mean()}")


def space(path):
    logging.info(f"path: {path}")

    logging.info("data loading...")
    columns = ['tid', 'time', 'lat', 'lon', 'stid']
    data1 = np.load(f"{path}data1.npy", allow_pickle=True)
    data1 = pd.DataFrame(data1, columns=columns).infer_objects()
    data1 = data1[columns[:-1]]

    logging.info("space...")
    lat1min, lat1max = data1.lat.min(), data1.lat.max()
    lon1min, lon1max = data1.lon.min(), data1.lon.max()
    logging.info(f"data1 trajectory lat min: {lat1min}, lat max: {lat1max}, lon min: {lon1min}, lon max: {lon1max}")
    data1['point'] = data1.apply(lambda row: (row.time, row.lat, row.lon), axis=1)
    data1 = data1.groupby("tid").agg({"point": list})
    data1 = data1[['point']]
    data1['point'] = data1['point'].map(lambda row: sorted(row, key=lambda tuples: tuples[0]))
    data1['point'] = data1['point'].map(
        lambda row: [geodesic(row[i - 1][1:], row[i][1:]).m for i in range(1, len(row))])
    data1['range'] = data1['point'].map(lambda row: np.sum(row))
    logging.info(f"data1 trajectory space range: {data1.range.sum()} m")
    data1['interval'] = data1['point'].map(lambda row: np.mean(row))
    logging.info(f"data1 trajectory space interval: {data1.interval.mean()} m")

    # data2
    data2 = np.load(f"{path}data2.npy", allow_pickle=True)
    data2 = pd.DataFrame(data2, columns=columns).infer_objects()
    data2 = data2[columns[:-1]]

    lat2min, lat2max = data2.lat.min(), data2.lat.max()
    lon2min, lon2max = data2.lon.min(), data2.lon.max()
    logging.info(f"data2 trajectory lat min: {lat2min}, lat max: {lat2max}, lon min: {lon2min}, lon max: {lon2max}")
    data2['point'] = data2.apply(lambda row: (row.time, row.lat, row.lon), axis=1)
    data2 = data2.groupby("tid").agg({"point": list})
    data2 = data2[['point']]
    data2['point'] = data2['point'].map(lambda row: sorted(row, key=lambda tuples: tuples[0]))
    data2['point'] = data2['point'].map(
        lambda row: [geodesic(row[i - 1][1:], row[i][1:]).m for i in range(1, len(row))])
    data2['range'] = data2['point'].map(lambda row: np.sum(row))
    logging.info(f"data2 trajectory space range: {data2.range.sum()} m")
    data2['interval'] = data2['point'].map(lambda row: np.mean(row))
    logging.info(f"data2 trajectory space interval: {data2.interval.mean()} m")


if __name__ == "__main__":
    logging.basicConfig(filename=f'./analysis.logs', format='%(asctime)s - %(message)s', level=logging.INFO)
    path = "./ais/"
    main(path)
    logging.info('-'*30)
    path = "./taxi/"
    main(path)

    logging.info('-' * 30)
    path = "./small_ais/"
    space(path)
    logging.info('-' * 30)
    path = "./small_taxi/"
    space(path)
