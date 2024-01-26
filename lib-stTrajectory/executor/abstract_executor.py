"""
Executor: executing model training and inference. The main functions include:

Model training: Input the prepared data into the model and train it.
    During training, techniques such as cross-validation can be used for model selection and tuning.
Model evaluation: Evaluation the trained model using the testing dataset.
    Calculate various metrics such as accuracy, precision, mean squared error, etc., to assess the model's performance.
Model saving and loading: Save the trained model to a file for future use.
"""


class AbstractExecutor(object):

    def __init__(self):
        pass

    def run(self, *args, **kwargs):
        pass
