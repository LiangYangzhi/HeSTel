import pandas as pd


class DataAugmentation(object):
    def __init__(self):
        pass

    def _appear_num_seg(self, times, num: list):
        """
        按出现次数进行划分
        :param times: columns as a series
        :param num: list
            int: 阈值次数，如果小于num则每次都切割，否则的话，是该参数的倍数才切割。
                每次的切割的时间长度内，轨迹点<=num
            list: in list 才会切割
        :return: [{}, {}, ...]
        """
        seg = []
        cursor = 0
        for i, t in enumerate(times):
            cursor += 1
            if cursor in num:
                dic = {"segment": cursor,
                       "start_time": times[0],
                       "end_time": t}
                seg.append(dic)
        return seg

    def data_segmentation(self, data: pd.DataFrame, col: list, num: list or int):
        """
        根据轨迹点个数切割
        :param data: pd.DataFrame
            trajectory data
        :param col: list
            The columns names corresponding to trajectory data, in order [user, time]
        :param num: int or list
            int: 阈值次数，如果小于等于num则每次都切割
                每次的切割的时间长度内，轨迹点<=num
            list: in list 才会切割
        :return: pd.DataFrame and columns,  columns = [user, segmentation, start_time, end_time]
        """
        if isinstance(num, int):
            num = [i for i in range(2, num + 1)]
        data = data[col]
        data = data.groupby(col[0]).agg({col[1]: list}).reset_index()  # col[0]=user, col[1]=time
        # segmentation
        data["seg"] = data.apply(lambda row: self._appear_num_seg(row[col[1]], num), axis=1)

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

    def non_segmentation(self, data: pd.DataFrame, col: list):
        """
        根据轨迹点个数切割
        :param data: pd.DataFrame
            trajectory data
        :param col: list
            The columns names corresponding to trajectory data, in order [user, time]
        :return: pd.DataFrame and columns,  columns = [user, segmentation, start_time, end_time]
        """
        data = data[col]
        data = data.groupby(col[0]).agg({col[1]: list}).reset_index()  # col[0]=user, col[1]=time
        # segmentation
        data["seg"] = data.apply(lambda row: self._max_seg(row[col[1]]), axis=1)

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

    def _max_seg(self, times):
        """
        按出现次数进行划分
        :param times: columns as a series
        :return: [{}, {}, ...]
        """
        if len(times) > 20:
            return None
        if len(times) < 1:
            return None
        seg = {"segment": "max",
               "start_time": times[0],
               "end_time": times[-1]}
        return seg
