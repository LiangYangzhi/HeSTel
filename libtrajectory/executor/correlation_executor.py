import lightgbm as lgb

from libtrajectory.executor.abstract_executor import AbstractExecutor


class LightGBMExecutor(AbstractExecutor):
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, *args, **kwargs):
        pass  # Todo

    def load_model(self, file):
        self.model = lgb.Booster(model_file=file)

    def save_model(self, file):
        self.model.save_model(file)
