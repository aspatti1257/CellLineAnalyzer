import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from ArgumentConfig.AnalysisType import AnalysisType
from ArgumentProcessingService import ArgumentProcessingService
from LoggerFactory import LoggerFactory
from Utilities.SafeCastUtil import SafeCastUtil


class DataFormattingService(object):

    log = LoggerFactory.createLog(__name__)
    
    TRAINING_MATRIX = "trainingMatrix"
    TESTING_MATRIX = "testingMatrix"  # Will either be outer testing or inner validation matrix

    P_VALUE_CUTOFF = 0.05

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self, should_scale, should_one_hot_encode=True):
        features_df = pd.DataFrame.from_dict(self.inputs.features, orient='index')
        columns = self.inputs.features.get(ArgumentProcessingService.FEATURE_NAMES)
        features_df.columns = columns
        features_df = features_df.drop(ArgumentProcessingService.FEATURE_NAMES)

        if should_one_hot_encode:
            one_hot_df = self.oneHot(features_df)
        else:
            one_hot_df = features_df

        x_train, x_test, y_train, y_test = self.testTrainSplit(one_hot_df, self.inputs.results, self.inputs.data_split)

        if self.inputs.analysisType() is AnalysisType.SPEARMAN_NO_GENE_LISTS:
            x_train_corr, x_test_corr = self.filterCorrelatedFeatures(x_train, x_test, columns, y_train)
        else:
            x_train_corr, x_test_corr = x_train, x_test

        outputs = OrderedDict()
        outputs[self.TRAINING_MATRIX] = self.maybeScaleFeatures(x_train_corr, should_scale)
        outputs[self.TESTING_MATRIX] = self.maybeScaleFeatures(x_test_corr, should_scale)
        outputs[ArgumentProcessingService.FEATURE_NAMES] = SafeCastUtil.safeCast(x_train_corr.columns, list)
        return outputs

    def maybeScaleFeatures(self, data_frame, should_scale):
        as_dict = data_frame.transpose().to_dict('list')
        maybe_scaled_dict = OrderedDict()

        keys_as_list = SafeCastUtil.safeCast(as_dict.keys(), list)
        for key in keys_as_list:
            maybe_scaled_dict[key] = []

        if len(keys_as_list) > 0:
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

    def filterCorrelatedFeatures(self, train_df, test_df, feature_names, train_results):
        results = [result[1] for result in train_results]
        filtered_train_df = train_df
        filtered_test_df = test_df

        cutoff_by_feature_length = self.P_VALUE_CUTOFF / len(feature_names)

        for feature_name in feature_names:
            try:
                spearman_corr = spearmanr(filtered_train_df.get(feature_name), results)
                p_val = SafeCastUtil.safeCast(spearman_corr[1], float, len(feature_names))

                if math.isnan(p_val) or p_val > cutoff_by_feature_length:
                    filtered_train_df = filtered_train_df.drop(feature_name, axis=1)
                    filtered_test_df = filtered_test_df.drop(feature_name, axis=1)
            except ValueError as error:
                self.log.error("Exception while trying to trim features: %s", error)

        return filtered_train_df, filtered_test_df
