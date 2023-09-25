from libtrajectory.model.abstract_model import AbstractModel
import lightgbm as lgb


class LightgbmModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model_params = None
        self.model = None

    def set_params(self, **params):
        self.model_params = params

    def train(self, X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        if self.model:  # model continue training
            self.model = lgb.train(self.model_params, train_data, init_model=self.model)
        else:  # first training
            self.model = lgb.train(self.model_params, train_data)
    def predict(self, X_test):
        return self.model.predict(X_test)

    def load_model(self, file):
        self.model = lgb.Booster(model_file=file)

    def save_model(self, file):
        self.model.save_model(file)
