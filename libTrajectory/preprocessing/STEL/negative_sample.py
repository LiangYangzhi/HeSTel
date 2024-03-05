import logging
from collections import Counter

import pandas as pd


class NegativeSample(object):
    def __init__(self, path):
        self.path = path
        self.data1 = None

    def run(self):
        self.data_load()
        self.generate_ns()

    def data_load(self):
        logging.info("data preparation...")
        dic = {'tid': str, 'time': int, 'lat': float, 'lon': float, 'did': str, 'spaceid': str, 'timeid': str,
               'stid': str}
        data1 = pd.read_csv(f"{self.path[: -4]}_train_data1.csv", dtype=dic)
        data2 = pd.read_csv(f"{self.path[: -4]}_train_data2.csv", dtype=dic)

        self.data1_group = data1.groupby('tid')
        self.data2_group = data2.groupby('tid')

        self.data1 = data1[['tid']].copy()
        self.data1.drop_duplicates(inplace=True)
        self.data2 = data2[['tid']].copy()
        self.data2.drop_duplicates(inplace=True)

        self.stid_tid1 = data1.groupby('stid').agg({"tid": set})
        self.stid_tid1['tid'] = self.stid_tid1['tid'].map(lambda x: list(x))
        self.stid_tid2 = data2.groupby('stid').agg({"tid": set})
        self.stid_tid2['tid'] = self.stid_tid2['tid'].map(lambda x: list(x))

        self.space_tid1 = data1.groupby('spaceid').agg({"tid": set})
        self.space_tid1['tid'] = self.space_tid1['tid'].map(lambda x: list(x))
        self.space_tid2 = data2.groupby('spaceid').agg({"tid": set})
        self.space_tid2['tid'] = self.space_tid2['tid'].map(lambda x: list(x))

        self.time_tid1 = data1.groupby('timeid').agg({"tid": set})
        self.time_tid1['tid'] = self.time_tid1['tid'].map(lambda x: list(x))
        self.time_tid2 = data2.groupby('timeid').agg({"tid": set})
        self.time_tid2['tid'] = self.time_tid2['tid'].map(lambda x: list(x))
        logging.info("data preparation completed")

    def generate_ns(self):
        logging.info("spatiotemporal negative sample1...")
        self.data1['st_ns_tid'] = self.data1.tid.map(lambda x: self.ns_tid(x, data='1', method='st'))
        self.data1.to_csv(f"{self.path[: -4]}_ns_sample1.csv", index=False)
        logging.info("spatial negative sample1...")
        self.data1['s_ns_tid'] = self.data1.tid.map(lambda x: self.ns_tid(x, data='1', method='s'))
        self.data1.to_csv(f"{self.path[: -4]}_ns_sample1.csv", index=False)
        logging.info("temporal negative sample1...")
        self.data1['t_ns_tid'] = self.data1.tid.map(lambda x: self.ns_tid(x, data='1', method='t'))
        self.data1.to_csv(f"{self.path[: -4]}_ns_sample1.csv", index=False)

        logging.info("spatiotemporal negative sample2...")
        self.data2['st_ns_tid'] = self.data2.tid.map(lambda x: self.ns_tid(x, data='2', method='st'))
        self.data2.to_csv(f"{self.path[: -4]}_ns_sample2.csv", index=False)
        logging.info("spatial negative sample1...")
        self.data2['s_ns_tid'] = self.data2.tid.map(lambda x: self.ns_tid(x, data='2', method='s'))
        self.data2.to_csv(f"{self.path[: -4]}_ns_sample2.csv", index=False)
        logging.info("temporal negative sample1...")
        self.data2['t_ns_tid'] = self.data2.tid.map(lambda x: self.ns_tid(x, data='2', method='t'))
        self.data2.to_csv(f"{self.path[: -4]}_ns_sample2.csv", index=False)

    def ns_tid(self, tid, data='1', method='st'):
        """
        正样本:A-B(tid相同), A的负样本从B中生成
        st: spatiotemporal
        s: spatial
        t: temporal
        ns: negative sample
        """
        print(tid)
        if data == '1':
            df = self.data1_group.get_group(tid).copy()
        elif data == '2':
            df = self.data2_group.get_group(tid).copy()
        else:
            raise ValueError(f"data={data} 不在方法random_enhance中。")

        if method == 'st':
            id_list = df.stid.unique().tolist()
            if data == '1':
                tid_lis = self.stid_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.stid_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")

        elif method == 's':
            id_list = df.spaceid.unique().tolist()
            if data == '1':
                tid_lis = self.space_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.space_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")

        elif method == 't':
            id_list = df.timeid.unique().tolist()
            if data == '1':
                tid_lis = self.time_tid1.loc[id_list, :].tid.tolist()
            elif data == '2':
                tid_lis = self.time_tid2.loc[id_list, :].tid.tolist()
            else:
                raise ValueError(f"data={data} 不在方法st_ns中。")
        else:
            raise ValueError(f"method={method} 不在方法st_ns中。")

        tid_lis = sum(tid_lis, [])
        most_counter = Counter(tid_lis).most_common(2)  # 出现最多的top2 tid
        if len(most_counter) == 1:
            return "None"
        if most_counter[0][0] == tid:
            ns_tid = most_counter[1][0]
        else:
            ns_tid = most_counter[0][0]
        return ns_tid

    def get(self):
        if self.data1 is None:
            logging.info("negative sample data loading...")
            dic = {'tid': str, 'st_ns_tid': str, 's_ns_tid': str, 't_ns_tid': str}
            self.data1 = pd.read_csv(f"{self.path[: -4]}_ns_sample1.csv", dtype=dic)
            self.data2 = pd.read_csv(f"{self.path[: -4]}_ns_sample2.csv", dtype=dic)
            logging.info("negative sample data load completed")
        return self.data1, self.data1

