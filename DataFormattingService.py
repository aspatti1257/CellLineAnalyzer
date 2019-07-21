import numpy as np
import pandas as pd
import math
import operator
from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.stats import ranksums
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
    NUM_TOP_FEATURES_TO_USE = 147

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self, should_scale, should_one_hot_encode=True):
        features_df = pd.DataFrame.from_dict(self.inputs.features, orient='index')
        columns = self.inputs.features.get(ArgumentProcessingService.FEATURE_NAMES)
        features_df.columns = columns
        features_df = features_df.drop(ArgumentProcessingService.FEATURE_NAMES)

        correlated_df = self.maybeFilterCorrelatedFeatures(features_df, columns, self.inputs.results,
                                                           self.inputs.analysisType())
        if should_one_hot_encode:
            one_hot_df = self.oneHot(correlated_df)
        else:
            one_hot_df = correlated_df

        x_train, x_test, y_train, y_test = self.testTrainSplit(one_hot_df, self.inputs.results, self.inputs.data_split)

        outputs = OrderedDict()
        outputs[self.TRAINING_MATRIX] = self.maybeScaleFeatures(x_train, should_scale)
        outputs[self.TESTING_MATRIX] = self.maybeScaleFeatures(x_test, should_scale)
        outputs[ArgumentProcessingService.FEATURE_NAMES] = SafeCastUtil.safeCast(x_train.columns, list)
        return outputs

    def maybeFilterCorrelatedFeatures(self, features_df, feature_names, labeled_results, analysis_type):
        if analysis_type is not AnalysisType.NO_GENE_LISTS:
            return features_df

        results = [result[1] for result in labeled_results]

        spearman_p_vals = {}
        ranksum_p_vals = {}

        for feature_name in feature_names:
            try:
                feature_column = features_df.get(feature_name)

                is_categorical = all(isinstance(feature, str) for feature in feature_column)
                file = feature_name.split(".")[0]

                if is_categorical:
                    if ranksum_p_vals.get(file) is None:
                        ranksum_p_vals[file] = {}
                    ranksum = self.fetchRanksum(feature_column, results)
                    ranksum_p_vals[file][feature_name] = SafeCastUtil.safeCast(ranksum[1], float, 1)
                else:
                    if spearman_p_vals.get(file) is None:
                        spearman_p_vals[file] = {}
                    spearman_corr = spearmanr(feature_column, results)
                    spearman_p_vals[file][feature_name] = SafeCastUtil.safeCast(spearman_corr[1], float, 1)

            except ValueError as error:
                self.log.error("Exception while trying to trim features: %s", error)

        return self.trimFeatures(features_df, [ranksum_p_vals, spearman_p_vals])

    def fetchRanksum(self, feature_column, results):
        value_counts = {}
        for val in feature_column:
            if value_counts.get(val) is None:
                value_counts[val] = 1
            else:
                value_counts[val] += 1
        dominant_value = max(value_counts.items(), key=operator.itemgetter(1))[0]
        dominant_results = []
        other_results = []
        for i in range(0, len(feature_column)):
            if feature_column[i] == dominant_value:
                dominant_results.append(results[i])
            else:
                other_results.append(results[i])
        return ranksums(dominant_results, other_results)

    def trimFeatures(self, features_df, p_val_sets):
        filtered_df = features_df
        for p_val_set in p_val_sets:
            for file in p_val_set:
                sorted_features = sorted(p_val_set[file].items(), key=operator.itemgetter(1))
                if len(sorted_features) > self.NUM_TOP_FEATURES_TO_USE:
                    for i in range(self.NUM_TOP_FEATURES_TO_USE, len(sorted_features)):
                        feature_to_drop = sorted_features[i][0]
                        try:
                            filtered_df = filtered_df.drop(feature_to_drop, axis=1)
                        except ValueError as error:
                            self.log.error("Unable to trim feature %s from dataframe: %s", feature_to_drop, error)
        return filtered_df

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
        return x_train, x_test, y_train, y_test

    def testStratifySplit(self, x_values, y_values):
        x_train, x_split, y_train, y_split = train_test_split(x_values, y_values, test_size=0.2, random_state=42,
                                                              stratify=x_values.iloc[:, -1])
        x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5, random_state=42,
                                                                  stratify=x_split.iloc[:, -1])
        return x_train,  x_validate, x_test, y_train, y_validate, y_test
