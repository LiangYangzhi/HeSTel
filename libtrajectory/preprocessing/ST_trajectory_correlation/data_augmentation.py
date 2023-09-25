import pandas as pd


class DataAugmentation(object):
    def __init__(self, data, col):
        """
        :param data: pd.DataFrame
            trajectory data
        :param col: list
            The columns names corresponding to trajectory data, in order [user, time]
        """
        self.data = data
        self.col = col

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

    def data_segmentation(self, num: list or int):
        """
        根据轨迹点个数切割
        :param num: int or list
            int: 阈值次数，如果小于等于num则每次都切割
                每次的切割的时间长度内，轨迹点<=num
            list: in list 才会切割
        :return: pd.DataFrame and columns,  columns = [user, segmentation, start_time, end_time]
        """
        if isinstance(num, int):
            num = [i for i in range(2, num + 1)]
        self.data = self.data[self.col]
        self.data = self.data.groupby(self.col["user"]).agg({self.col['time']: list}).reset_index()
        # segmentation
        self.data["seg"] = self.data.apply(lambda row: self._appear_num_seg(row[self.col['time']], num), axis=1)

        segment = self.data.explode("seg")
        segment = segment.dropna()
        segment = pd.concat([segment, segment['seg'].apply(pd.Series)], axis=1).drop('seg', axis=1)
        col_segment = {"user1": self.col["user"],
                       "segment": "segment",
                       "start_time": "start_time",
                       "end_time": "end_time"}
        segment = segment[list(col_segment.values())]
        return segment, col_segment

    def max_length(self):
        """
        根据轨迹点个数切割
        """
        self.data = self.data.groupby(self.col["user"]).agg({self.col['time']: list}).reset_index()
        # segmentation
        self.data["seg"] = self.data.apply(lambda row: self._max_seg(row[self.col['time']]), axis=1)

        segment = self.data.dropna()
        segment = pd.concat([segment, segment['seg'].apply(pd.Series)], axis=1).drop('seg', axis=1)
        col_segment = {"user1": self.col["user"],
                       "segment": "segment",
                       "start_time": "start_time",
                       "end_time": "end_time"}
        segment = segment[list(col_segment.values())]
        return segment, col_segment

    def _max_seg(self, times):
        """
        按出现次数进行划分
        :param times: columns as a series
        :return: dict
        """
        # if len(times) > 20:
        #     seg = {"segment": "max",
        #            "start_time": times[0],
        #            "end_time": times[20]}
        #     return seg
        if len(times) < 1:
            return None
        seg = {"segment": len(times),
               "start_time": times[0],
               "end_time": times[-1]}
        return seg
