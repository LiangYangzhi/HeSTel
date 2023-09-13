import pandas as pd


class DataAugmentation(object):
    def __init__(self):
        pass

    def _appear_num_seg(self, times, num: int, no_num=None, max_num=None):
        """
        按出现次数进行划分
        :param times: columns as a series
        :param num: int
            阈值次数，如果小于num则每次都切割，否则的话，是该参数的倍数才切割。
            每次的切割的时间长度内，轨迹点<=num
        :param no_num: None or list
            None: 满足条件的次数都进行切割。
            list: 满足条件 & 出现次数不在list中，才会切割。
        :param max_num: None or int
            None: 满足条件的次数都进行切割。
            list: 满足条件 & 不大于max_num，才会切割。
        :return: [{}, {}, ...]
        """
        if no_num is None:
            no_num = []
        seg = []
        cursor = 0
        for i, t in enumerate(times):
            cursor += 1
            if cursor <= num and cursor not in no_num:
                dic = {"segment": cursor,
                       "start_time": times[0],
                       "end_time": t}
                seg.append(dic)

            elif cursor % num == 0 and cursor not in no_num:
                if max_num is None or cursor < max_num:
                    dic = {
                        "segment": cursor,
                        "start_time": times[cursor - num],
                        "end_time": t}
                    seg.append(dic)
        return seg

    def data_segmentation(self, data: pd.DataFrame, col: list, num: int, no_num=None, max_num=None):
        """
        根据轨迹点个数切割
        :param data: pd.DataFrame
            trajectory data
        :param col: list
            The columns names corresponding to trajectory data, in order [user, time]
        :param num: int
            阈值次数，如果小于num则每次都切割，否则的话，是该参数的倍数才切割。
            每次的切割的时间长度内，轨迹点<=num
        :param no_num: None or list
            None: 满足条件的次数都进行切割。
            list: 满足条件 & 出现次数不在list中，才会切割。
        :param max_num: None or int
            None: 满足条件的次数都进行切割。
            list: 满足条件 & 不大于max_num，才会切割。
        :return: pd.DataFrame and columns,  columns = [user, segmentation, start_time, end_time]
        """
        data = data[col]
        data = data.groupby(col[0]).agg({col[1]: list}).reset_index()  # col[0]=user, col[1]=time
        # segmentation
        data["seg"] = data.apply(lambda row: self._appear_num_seg(row[col[1]], num, no_num, max_num), axis=1)

        data = data.explode("seg")
        data = data.dropna()
        data["segment"] = data.seg.map(lambda row: row['segment'])
        data["start_time"] = data.seg.map(lambda row: row['start_time'])
        data["end_time"] = data.seg.map(lambda row: row['end_time'])
        col_segmentation = {"user1": col[0],
                            "segment": "segment",
                            "start_time": "start_time",
                            "end_time": "end_time"}
        data = data[list(col_segmentation.values())]
        return data, col_segmentation
