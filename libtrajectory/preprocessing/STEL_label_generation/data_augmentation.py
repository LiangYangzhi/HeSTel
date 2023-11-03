import pandas as pd


class DataAugmentation(object):
    def __init__(self, data, col):
        self.data = data
        self.col = col

    def no_augmentation(self):
        """
        no trajectory augmentation
        :return:
        """
        self.data = self.data.groupby(self.col["user"]).agg({self.col['time']: list}).reset_index()
        # segmentation
        self.data["seg"] = self.data.apply(
            lambda row: self._record_time(row[self.col['time']]), axis=1)

        segment = self.data.dropna()
        segment = pd.concat([segment, segment['seg'].apply(pd.Series)], axis=1).drop('seg', axis=1)
        col_segment = {"user1": self.col["user"],
                       "segment": "segment",
                       "start_time": "start_time",
                       "end_time": "end_time"}
        segment = segment[list(col_segment.values())]
        return segment, col_segment

    def _record_time(self, times):
        if times.__len__() < 1:
            return None
        return {"segment": "max",
                "start_time": times[0],
                "end_time": times[-1]}
