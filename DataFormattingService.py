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

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = {"features": {}}

    def formatData(self):
        features = self.inputs.get(ArgumentProcessingService.FEATURES)
        feature_names = features.get(ArgumentProcessingService.FEATURE_NAMES)
        self.outputs["features"]["featureNames"] = self.inputs["features"]["featureNames"]
        self.outputs["results"] = self.inputs["results"]
        self.outputs["is_classifier"] = self.inputs["is_classifier"]
        features_df = pd.DataFrame.from_dict(self.inputs["features"], orient='index')
        features_df = features_df.drop("featureNames")
        features_oh_df = self.oneHot(features_df)
        X_train, X_validate, X_test, y_train, y_validate, y_test = self.testTrainSplit(features_oh_df, self.inputs["results"])
        self.outputs["trainingMatrix"] = X_train.transpose().to_dict('list')
        self.outputs["testingMatrix"] = X_test.transpose().to_dict('list')
        self.outputs["validationMatrix"] = X_validate.transpose().to_dict('list')
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

    def testTrainSplit(self, X_values, y_values):
        X_train, X_split, y_train, y_split = train_test_split(X_values, y_values, test_size=0.2, random_state=42)
        X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=42)
        return X_train, X_validate, X_test, y_train, y_validate, y_test

    def testStratifySplit(self, X_values, y_values):
        X_train, X_split, y_train, y_split = train_test_split(X_values, y_values, test_size=0.2, random_state=42,
                                                              stratify=X_values.iloc[:, -1])
        X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=42,
                                                                  stratify=X_split.iloc[:, -1])
        return X_train,  X_validate, X_test, y_train, y_validate, y_test