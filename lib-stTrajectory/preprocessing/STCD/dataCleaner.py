import pandas as pd


class Cleaner(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def filter_place(self, min_num: int, user_col='userid', place_col='placeid'):
        print(f"---before filter user number: {self.data[user_col].unique().__len__()}, "
              f"place number: {self.data[place_col]}")
        col = [user_col, place_col]
        data = self.data[col]

        place_freq = data.groupby(place_col).count()
        place_freq.reset_index(inplace=True)
        place = place_freq.query(f"{user_col} >= {min_num}")[place_col].unique().tolist()
        self.data = self.data.query(f"{user_col} in {place}")

        print(f"--after filter user number: {self.data[user_col].unique().__len__()}, "
              f"place number: {place.__len__()}")

    def get_data(self):
        return self.data
