from libtrajectory.executor.abstract_executor import AbstractExecutor
from libtrajectory.model.lightgbm_model import LightgbmModel


class LightGBMExecutor(AbstractExecutor):
    def __init__(self, X_train=None, y_train=None, X_test=None, config=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.config = config
        self.model = LightgbmModel()

    def run(self):
        if self.config.get("model_file", None):
            print("model load")
            self._load_model()
        if self.config.get("train", None):
            print("model train")
            self._train()
        print("model predict")
        return self._predict()

    def save_model(self):
        self.model.save_model(self.config['save_model'])

    def _load_model(self):
        self.model.load_model(self.config['model_file'])

    def _train(self):
        self.model.set_params(**self.config['params'])
        self.model.train(X_train=self.X_train, y_train=self.y_train)

    def _predict(self):
        return self.model.predict(X_test=self.X_test)
