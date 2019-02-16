import numpy as np
import pandas as pd
import math
import logging
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil


class DataFormattingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)
    
    TRAINING_MATRIX = "trainingMatrix"
    TESTING_MATRIX = "testingMatrix"  # Will either be outer testing or inner validation matrix

    P_VALUE_CUTOFF = 0.05

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self, should_scale, should_one_hot_encode=True):
        features_df = pd.DataFrame.from_dict(self.inputs[ArgumentProcessingService.FEATURES], orient='index')
        columns = self.inputs.get(ArgumentProcessingService.FEATURES).get(ArgumentProcessingService.FEATURE_NAMES)
        features_df.columns = columns
        features_df = features_df.drop(ArgumentProcessingService.FEATURE_NAMES)

        if should_one_hot_encode:
            features_oh_df = self.oneHot(features_df)
        else:
            features_oh_df = features_df

        if self.inputs.get(ArgumentProcessingService.SPEARMAN_CORR):
            correlated_df = self.filterCorrelatedFeatures(features_oh_df, columns)
        else:
            correlated_df = features_oh_df

        x_train, x_test, y_train, y_test = self.testTrainSplit(correlated_df,
                                                               self.inputs[ArgumentProcessingService.RESULTS],
                                                               self.inputs[ArgumentProcessingService.DATA_SPLIT])

        outputs = OrderedDict()
        outputs[self.TRAINING_MATRIX] = self.maybeScaleFeatures(x_train, should_scale)
        outputs[self.TESTING_MATRIX] = self.maybeScaleFeatures(x_test, should_scale)
        outputs[ArgumentProcessingService.FEATURE_NAMES] = SafeCastUtil.safeCast(correlated_df.columns, list)
        return outputs

    def maybeScaleFeatures(self, data_frame, should_scale):
        as_dict = data_frame.transpose().to_dict('list')
        maybe_scaled_dict = OrderedDict()

        keys_as_list = SafeCastUtil.safeCast(as_dict.keys(), list)
        for key in keys_as_list:
            maybe_scaled_dict[key] = []

        for i in range(0, len(as_dict[keys_as_list[0]])):
            array_to_maybe_scale = []
            for key in keys_as_list:
                array_to_maybe_scale.append(as_dict[key][i])
            if should_scale:
                maybe_scaled_array = preprocessing.scale(array_to_maybe_scale)
            else:
                maybe_scaled_array = array_to_maybe_scale
            for j in range(0, len(keys_as_list)):
                maybe_scaled_dict[keys_as_list[j]].append(maybe_scaled_array[j])

        return maybe_scaled_dict

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

    def filterCorrelatedFeatures(self, df, feature_names):
        results = [result[1] for result in self.inputs.get(ArgumentProcessingService.RESULTS)]
        filtered_df = df

        for feature_name in feature_names:
            spearman_corr = spearmanr(filtered_df.get(feature_name), results)
            p_val = spearman_corr[1]

            if math.isnan(p_val) or (p_val / len(feature_names)) > self.P_VALUE_CUTOFF:
                filtered_df = filtered_df.drop(feature_name, axis=1)

        return filtered_df
