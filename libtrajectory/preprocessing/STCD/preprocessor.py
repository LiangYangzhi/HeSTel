from datetime import datetime
from gensim.models import Word2Vec

from libtrajectory.preprocessing.STCD.dataCleaner import Cleaner
from libtrajectory.preprocessing.STCD.dataLoader import Loader
from libtrajectory.preprocessing.STCD.featureExtraction import Extractor
from libtrajectory.preprocessing.STCD.generateCorpus import Generator as Generator_corpus
from libtrajectory.preprocessing.STCD.generateHypergraph import Generator as Generator_hypergraph
from libtrajectory.preprocessing.abstract_preprocessor import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    def __init__(self, config):
        self.temporal_edge = None
        self.spatial_edge = None
        self.node = None
        self.spatial_model = None
        self.placeid_model = None
        self.label = None
        self.data = None
        self.config = config

    def data_loading(self, sample):
        print("data loading")
        t0 = datetime.now()
        loader = Loader()
        self.data = loader.get_data(**self.config['data'])
        self.label = loader.get_data(**self.config['label'])
        if sample:
            self.data = self.data.sample(sample, random_state=23742)
        print(f"data1 number : {self.data.shape}")
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def data_cleaning(self):
        print("data cleaning")
        t0 = datetime.now()
        config = self.config['clean']
        cleaner = Cleaner(self.data)
        cleaner.filter_place(**config['filter_place'])
        self.data = cleaner.get_data()
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def feature_extraction(self):
        print("data cleaning")
        t0 = datetime.now()
        config = self.config['feature']
        extractor = Extractor(self.data)
        extractor.temporal_feature(config['time_col'])
        extractor.spatio_feature(config['lat_col'], config['lon_col'])
        self.data = extractor.get_data()
        print(self.data.info())
        print(self.data.head())
        print(self.data.groupby('userid').agg("count")[config['time_col']].describe())
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def pretrian(self):
        print("pretraining")
        t0 = datetime.now()
        config = self.config['feature']
        generator = Generator_corpus(self.data)
        corpus1 = generator.user_place(**config['user_place']['corpus'])
        self.placeid_model = Word2Vec(sentences=corpus1, **config['user_place']['model'])
        corpus2 = generator.place_lon_lat(**config['place_lon_lat']['corpus'])
        self.spatial_model = Word2Vec(sentences=corpus2, **config['place_lon_lat']['model'])
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def generate_hypergraph(self):
        print("generate hypergraph...")
        t0 = datetime.now()
        config = self.config['hypergraph']
        generator = Generator_hypergraph(self.data)
        generator.hypergraph(**config)
        self.node, self.spatial_edge, self.temporal_edge = generator.get_data()
        print(f'Running time: {datetime.now() - t0} Seconds', '\n')

    def get_data(self):
        return self.data, self.label, self.placeid_model, self.spatial_model, self.node, self.spatial_edge, self.temporal_edge
