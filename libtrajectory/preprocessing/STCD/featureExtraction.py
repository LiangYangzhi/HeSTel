import pandas as pd


class Extractor(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def temporal_feature(self, time_col="time"):
        print("--extracting time feature")
        self.data[time_col] = pd.to_datetime(self.data[time_col])
        self.data['hour'] = self.data[time_col].dt.hour  # [0~23], 按小时表示几点
        self.data['weekday'] = self.data[time_col].dt.day_of_week  # [0~6], 0表示周一
        self.data['month'] = self.data[time_col].dt.month
        self.data['week_of_year'] = self.data[time_col].dt.weekofyear
        self.data['first_day_of_month'] = self.data[time_col].map(lambda x: x.replace(day=1))
        self.data['week_of_month'] = self.data['week_of_year'] - self.data['first_day_of_month'].dt.weekofyear
        self.data['day_of_year'] = self.data[time_col].dt.dayofyear
        self.data['day_of_month'] = self.data[time_col].dt.day
        self.data['season'] = self.data[time_col].dt.quarter
        self.data['timestamp'] = self.data[time_col].apply(lambda x: int(x.value / 10 ** 9))
        self.data = self.data.sort_values('timestamp')
        print(f'temporal modelling complete')

    def spatio_feature(self, lat_col='lat', lon_col='lon'):
        print("--extracting spatio feature")
        min_lat = self.data[lat_col].min()
        max_lat = self.data[lat_col].max()
        min_lon = self.data[lon_col].min()
        max_lon = self.data[lon_col].max()
        range_lat = max_lat - min_lat
        range_lon = max_lon - min_lat

        self.data['normalize_lat'] = (self.data[lat_col] - min_lat) / range_lat
        self.data['normalize_lon'] = (self.data[lon_col] - min_lon) / range_lon

    def get_data(self, path=None):
        if path:
            self.data = pd.read_csv(path)
        return self.data
