import unittest
import numpy as np
import os
import logging
import pandas as pd
<<<<<<< HEAD

from DataFormattingService import DataFormattingService
=======

from DataFormattingService import DataFormattingService

>>>>>>> 3c94482215b7fbafca0dfbbe68bcef6408540ab3

class DataFormattingServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

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
<<<<<<< HEAD
        s = DataFormattingService(object)
        categorical_pd = pd.read_csv('SampleClassifierDataFolder/categorical.csv', delimiter=',')
        assert ((s.binary_one_hot(categorical_pd).dtypes.values != np.dtype('float64')).all() == True)
        assert ((s.one_hot(categorical_pd).dtypes.values != np.dtype('float64')).all() == True)

    def testsplit(self):
        s = DataFormattingService(object)
        features = pd.read_csv('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = pd.read_csv('SampleClassifierDataFolder/results.csv', delimiter=',')
        X_train, X_validate, X_test, y_train, y_validate, y_test = s.test_train_split(features, results)
        assert (len(X_train) and len(X_validate) and len(X_test) and len(y_train) and len(y_validate) and len(y_test) != 0)

    def stratify_split(self):
        s = DataFormattingService(object)
        features = pd.read_csv('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = pd.read_csv('SampleClassifierDataFolder/results.csv', delimiter=',')
        X_train, X_validate, X_test, y_train, y_validate, y_test = s.test_train_split(features, results)
        assert (len(X_train) and len(X_validate) and len(X_test) and len(y_train) and len(y_validate) and len(y_test) != 0)
=======
        categorical_pd = pd.read_csv(self.current_working_dir +
                                     '/SampleClassifierDataFolder/categorical.csv', delimiter=',')
        data_formatting_service = DataFormattingService(None)
        assert type(data_formatting_service.oneHot(categorical_pd)) == tuple
>>>>>>> 3c94482215b7fbafca0dfbbe68bcef6408540ab3


