import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging

class DataFormattingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self):
        pass

    def encode_categorical(self, array):
        if not array.dtype == np.dtype('float64') or array.dtype == np.dtype('int') :
            return preprocessing.LabelEncoder().fit_transform(array)
        else:
            return array

    # Encode sites as categorical variables
    def one_hot (self, dataframe):
        # Encode all labels
        dataframe = dataframe.apply(self.encode_categorical)
        return dataframe

    # Binary one hot encoding
    def binary_one_hot(self, dataframe):
        dataframe_binary_pd = pd.get_dummies(dataframe)
        return dataframe_binary_pd

    def test_train_split(self, X_values, y_values):
        X_train, X_split, y_train, y_split = train_test_split(X_values, y_values, test_size = 0.2, random_state = 42)
        X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=42)
        return X_train, X_validate, X_test, y_train, y_validate, y_test

    def stratify_split(self, X_values, y_values):
        X_train, X_split, y_train, y_split = train_test_split(X_values, y_values, test_size=0.2, random_state=42, stratify = X_values.iloc[:,-1])
        X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=42, stratify = X_split.iloc[:,-1])
        return X_train,  X_validate, X_test, y_train, y_validate, y_test

s = DataFormattingService(object)
categorical = pd.read_csv('Testing/SampleClassifierDataFolder/categorical.csv', delimiter=',')
features = pd.read_csv('Testing/SampleClassifierDataFolder/features.csv', delimiter=',')
results = pd.read_csv('Testing/SampleClassifierDataFolder/results.csv', delimiter=',')
print(s.one_hot(categorical))
# print(s.stratify_split(features, results))
