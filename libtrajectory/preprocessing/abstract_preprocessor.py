"""
Preprocessor: preprocessing data. The main functions include:

Data loading：Reading raw data from the file
Data cleaning：Removing erroneous data such as missing values and anomalies
Feature engineering：Extracting and transforming data to generate features that can be used for modeling.
Data splitting: It can split the dataset into training, validation, and testing sets.
Data normalization: Standardizing data to improve the training effectiveness of model
    Todo(Deep learning data normalization)
"""
from abc import ABC


class AbstractPreprocessor(ABC):
    def __init__(self):
        pass
