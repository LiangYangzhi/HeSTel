import pandas as pd


class Loader(object):
    def __init__(self):
        self.rename = None
        self.path = None
        self.data = None

    def get_data(self, path, rename):
        self.path = path
        self.rename = rename
        self.data = self._load_txt()
        self._rename()
        return self.data

    def _load_txt(self):
        df = pd.read_csv(self.path, header=None, sep='\t')
        return df

    def _rename(self):
        if isinstance(self.rename, str):
            self.rename = eval(self.rename)
        self.data.rename(columns=self.rename, inplace=True)
