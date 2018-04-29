import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging

from collections import OrderedDict
from ArgumentProcessingService import ArgumentProcessingService
from Utilities import SafeCastUtil


class DataFormattingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)
    
    TRAINING_MATRIX = "trainingMatrix"
    TESTING_MATRIX = "testingMatrix"  # Will either be outer testing or inner validation matrix

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = OrderedDict()

    def formatData(self):
        features_df = pd.DataFrame.from_dict(self.inputs[ArgumentProcessingService.FEATURES], orient='index')
        features_df = features_df.drop(ArgumentProcessingService.FEATURE_NAMES)
        features_oh_df = self.oneHot(features_df)

        x_train, x_test, y_train, y_test = self.testTrainSplit(features_oh_df,
                                                               self.inputs[ArgumentProcessingService.RESULTS],
                                                               self.inputs[ArgumentProcessingService.DATA_SPLIT])

        self.outputs[self.TRAINING_MATRIX] = self.scaleFeatures(x_train)
        self.outputs[self.TESTING_MATRIX] = self.scaleFeatures(x_test)
        return self.outputs

    def scaleFeatures(self, data_frame):
        as_dict = data_frame.transpose().to_dict('list')
        scaled_dict = OrderedDict()

        keys_as_list = SafeCastUtil.SafeCastUtil.safeCast(as_dict.keys(), list)
        for key in keys_as_list:
            scaled_dict[key] = []

        for i in range(0, len(as_dict[keys_as_list[0]])):
            array_to_scale = []
            for key in keys_as_list:
                array_to_scale.append(as_dict[key][i])
            scaled_array = preprocessing.scale(array_to_scale)
            for j in range(0, len(keys_as_list)):
                scaled_dict[keys_as_list[j]].append(scaled_array[j])

        return scaled_dict

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

    def testTrainSplit(self, x_values, y_values, data_split):
        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=(1 - data_split))
        # x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5)
        return x_train, x_test, y_train, y_test

    def testStratifySplit(self, x_values, y_values):
        x_train, x_split, y_train, y_split = train_test_split(x_values, y_values, test_size=0.2, random_state=42,
                                                              stratify=x_values.iloc[:, -1])
        x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5, random_state=42,
                                                                  stratify=x_split.iloc[:, -1])
        return x_train,  x_validate, x_test, y_train, y_validate, y_test
