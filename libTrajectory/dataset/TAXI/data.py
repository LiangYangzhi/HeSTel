import os

import pandas as pd
from tqdm import tqdm


class ProTra(object):
    def __init__(self):
        pass

    def trajectory(self):
        files = os.listdir('./taxi')
        colunms = ['uid', 'time', 'lon', 'lat']
        for f in tqdm(files):
            df = pd.read_csv(f,sep='\t', header=None,names=colunms)


