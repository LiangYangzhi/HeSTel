"""
Models: model definition. The main functions include:

Model definition: Choose an appropriate model based on the task type and data characteristics.
    During model definition, set the model's hyperparameters and optimizer(Todo).
Fit: model training
Predict: model prediction
"""


class AbstractModel(object):
    def __init__(self):
        self.model = None

