from libtrajectory.preprocessing.STCD.preprocessor import Preprocessor


def pipeline(config):
    preprocessor = Preprocessor(config['preprocessing'])
    preprocessor.data_loading(sample=config['sample'])
    preprocessor.data_cleaning()
    preprocessor.feature_extraction()
    preprocessor.pretrian()
    preprocessor.generate_hypergraph()
    data, label, placeid_model, spatial_model, node, spatial_edge, temporal_edge = preprocessor.get_data()

