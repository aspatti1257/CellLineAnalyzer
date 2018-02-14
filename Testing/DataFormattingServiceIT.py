import unittest
import numpy as np
import os
import logging
import pandas as pd
import math

from DataFormattingService import DataFormattingService
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil


class DataFormattingServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        self.instantiateDataFormattingService(input_folder)

    def tearDown(self):
        pass

    def instantiateDataFormattingService(self, input_folder):
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        self.data_formatting_service = DataFormattingService(arguments)

    def testFormattingDataRandomizesMatrices(self):
        original_outputs = self.data_formatting_service.formatData()
        self.validateOutput(original_outputs)

        self.instantiateDataFormattingService(self.current_working_dir + "/SampleClassifierDataFolder")
        new_outputs = self.data_formatting_service.formatData()
        self.validateOutput(new_outputs)

        original_trained_cells = SafeCastUtil.safeCast(original_outputs.get(DataFormattingService.TRAINING_MATRIX).keys(), list)
        new_trained_cells = SafeCastUtil.safeCast(new_outputs.get(DataFormattingService.TRAINING_MATRIX).keys(), list)
        non_identical_matrices = False
        for i in range(0, len(new_trained_cells)):
            if original_trained_cells[i] != new_trained_cells[i]:
                non_identical_matrices = True
        assert non_identical_matrices

    @staticmethod
    def validateOutput(output):
        assert output is not None
        assert output.get(DataFormattingService.TRAINING_MATRIX) is not None
        assert output.get(DataFormattingService.TESTING_MATRIX) is not None
        assert output.get(DataFormattingService.VALIDATION_MATRIX) is not None
        num_train = len(output.get(DataFormattingService.TRAINING_MATRIX).keys())
        num_test = len(output.get(DataFormattingService.TESTING_MATRIX).keys())
        num_val = len(output.get(DataFormattingService.VALIDATION_MATRIX).keys())
        assert math.isclose(num_val, num_test, abs_tol=1.0)
        assert num_train > (num_test + num_val)

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
        assert ((s.binaryOneHot(categorical_pd).dtypes.values != np.dtype('float64')).all())
        assert ((s.oneHot(categorical_pd).dtypes.values != np.dtype('float64')).all())

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
        categorical_onehot = data_formatting_service.oneHot(categorical_pd)
        assert (np.shape(categorical_onehot))[1] == 2



