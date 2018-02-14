import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging

from ArgumentProcessingService import ArgumentProcessingService


class DataFormattingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)
    
    TRAINING_MATRIX = "trainingMatrix"
    TESTING_MATRIX = "testingMatrix"
    VALIDATION_MATRIX = "validationMatrix"

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = {}

    def formatData(self):
        features = self.inputs.get(ArgumentProcessingService.FEATURES)
        feature_names = features.get(ArgumentProcessingService.FEATURE_NAMES)
        self.outputs[ArgumentProcessingService.FEATURE_NAMES] = feature_names
        self.outputs[ArgumentProcessingService.RESULTS] = self.inputs[ArgumentProcessingService.RESULTS]
        self.outputs[ArgumentProcessingService.IS_CLASSIFIER] = self.inputs[ArgumentProcessingService.IS_CLASSIFIER]

        features_df = pd.DataFrame.from_dict(self.inputs[ArgumentProcessingService.FEATURES], orient='index')
        features_df = features_df.drop(ArgumentProcessingService.FEATURE_NAMES)
        features_oh_df = self.oneHot(features_df)
        
        x_train, x_validate, x_test, y_train, y_validate, y_test = \
            self.testTrainSplit(features_oh_df, self.inputs[ArgumentProcessingService.RESULTS])
        
        self.outputs[self.TRAINING_MATRIX] = x_train.transpose().to_dict('list')
        self.outputs[self.TESTING_MATRIX] = x_test.transpose().to_dict('list')
        self.outputs[self.VALIDATION_MATRIX] = x_validate.transpose().to_dict('list')
        return self.outputs

    def encodeCategorical(self, array):
        if array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64'):
            return array
        else:
            return preprocessing.LabelEncoder().fit_transform(array)

    # Encode sites as categorical variables
    def oneHot(self, dataframe):
        # Encode all labels
        dataframe = dataframe.apply(self.encodeCategorical)
        return dataframe

    # Binary one hot encoding
    def binaryOneHot(self, dataframe):

        dataframe_binary_pd = pd.get_dummies(dataframe)
        return dataframe_binary_pd

    def testTrainSplit(self, x_values, y_values):
        x_train, x_split, y_train, y_split = train_test_split(x_values, y_values, test_size=0.2)
        x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5)
        return x_train, x_validate, x_test, y_train, y_validate, y_test

    def testStratifySplit(self, x_values, y_values):
        x_train, x_split, y_train, y_split = train_test_split(x_values, y_values, test_size=0.2, random_state=42,
                                                              stratify=x_values.iloc[:, -1])
        x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5, random_state=42,
                                                                  stratify=x_split.iloc[:, -1])
        return x_train,  x_validate, x_test, y_train, y_validate, y_test
