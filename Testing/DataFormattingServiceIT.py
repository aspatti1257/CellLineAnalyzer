import unittest
import numpy as np
import os
import logging
import pandas as pd

from DataFormattingService import DataFormattingService
from ArgumentProcessingService import ArgumentProcessingService


class DataFormattingServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        self.data_formatting_service = DataFormattingService(arguments)

    def tearDown(self):
        pass

    def testCheckImportData(self):
        features = np.genfromtxt(self.current_working_dir +
                                 '/SampleClassifierDataFolder/features.csv', delimiter=',')
        results = np.genfromtxt(self.current_working_dir +
                                '/SampleClassifierDataFolder/results.csv', delimiter=',')
        assert np.array(features[1:]).dtype == "float64"
        assert np.array(results[1:, 1]).dtype == "float64"
        assert not np.isnan(features[1:]).any()
        assert not np.isnan(results[1:, 1]).any()
        assert len(features) == len(results)

    def testCheckOneHotEncoding(self):
        s = self.data_formatting_service
        categorical_pd = pd.read_csv('SampleClassifierDataFolder/categorical.csv', delimiter=',')
        assert ((s.binaryOneHot(categorical_pd).dtypes.values != np.dtype('float64')).all() == True)
        assert ((s.oneHot(categorical_pd).dtypes.values != np.dtype('float64')).all() == True)

    def testSplit(self):
        s = self.data_formatting_service
        features = pd.read_csv('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = pd.read_csv('SampleClassifierDataFolder/results.csv', delimiter=',')
        X_train, X_validate, X_test, y_train, y_validate, y_test = s.testTrainSplit(features, results)
        assert (len(X_train) and len(X_validate) and len(X_test) and len(y_train) and len(y_validate) and len(y_test) != 0)

    def testStratifySplit(self):
        s = self.data_formatting_service
        features = pd.read_csv('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = pd.read_csv('SampleClassifierDataFolder/results.csv', delimiter=',')
        X_train, X_validate, X_test, y_train, y_validate, y_test = s.testTrainSplit(features, results)
        assert (len(X_train) and len(X_validate) and len(X_test) and len(y_train) and len(y_validate) and len(y_test) != 0)

        categorical_pd = pd.read_csv(self.current_working_dir +
                                     '/SampleClassifierDataFolder/categorical.csv', delimiter=',')
        data_formatting_service = DataFormattingService(None)
        assert type(data_formatting_service.oneHot(categorical_pd)) == tuple



