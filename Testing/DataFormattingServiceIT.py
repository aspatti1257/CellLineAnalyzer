import unittest
import numpy as np
import os
import pandas as pd

from DataFormattingService import DataFormattingService
from ArgumentProcessingService import ArgumentProcessingService
from LoggerFactory import LoggerFactory
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil


class DataFormattingServiceIT(unittest.TestCase):

    log = LoggerFactory.createLog(__name__)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        self.instantiateDataFormattingService(input_folder)

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                if file == "__init__.py":
                    continue
                os.remove(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def instantiateDataFormattingService(self, input_folder):
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        self.data_formatting_service = DataFormattingService(arguments)

    def fetchTrainAndTestData(self):
        s = self.data_formatting_service
        features = pd.read_csv('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = pd.read_csv('SampleClassifierDataFolder/results.csv', delimiter=',')
        X_train, X_test, y_train, y_test = s.testTrainSplit(features, results,
                                                            self.data_formatting_service.inputs.data_split)
        return X_test, X_train, y_test, y_train

    def testFormattingDataRandomizesMatrices(self):
        original_outputs = self.data_formatting_service.formatData(True)
        self.validateOutput(original_outputs)

        self.instantiateDataFormattingService(self.current_working_dir + "/SampleClassifierDataFolder")
        new_outputs = self.data_formatting_service.formatData(True)
        self.validateOutput(new_outputs)

        original_trained_cells = SafeCastUtil.safeCast(original_outputs.get(DataFormattingService.TRAINING_MATRIX).keys(), list)
        new_trained_cells = SafeCastUtil.safeCast(new_outputs.get(DataFormattingService.TRAINING_MATRIX).keys(), list)
        non_identical_matrices = False
        for i in range(0, len(new_trained_cells)):
            if original_trained_cells[i] != new_trained_cells[i]:
                non_identical_matrices = True
        assert non_identical_matrices

    def testFormattingRandomizedData(self):
        self.validateOutput(self.formatRandomizedData(True))
        self.validateOutput(self.formatRandomizedData(False))

    def formatRandomizedData(self, is_classifier):
        arguments = self.processArguments(is_classifier)
        data_formatting_service = DataFormattingService(arguments)
        return data_formatting_service.formatData(True)

    def processArguments(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(5, 50, 150, is_classifier, 10, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        return arguments

    def testSpearmanRTrimmingDoesNotTrimSignificantFeatures(self):
        significant_prefix = RandomizedDataGenerator.SIGNIFICANT_FEATURE_PREFIX
        arguments = self.processArguments(True)
        arguments.analyze_all = True
        orig_features = arguments.features.get(ArgumentProcessingService.FEATURE_NAMES)
        orig_sig_features = [feature for feature in orig_features if significant_prefix in feature]
        data_formatting_service = DataFormattingService(arguments)
        output = data_formatting_service.formatData(True)
        trimmed_features = output.get(ArgumentProcessingService.FEATURE_NAMES)
        trimmed_sig_features = [feature for feature in trimmed_features if significant_prefix in feature]

        assert len(orig_features) > len(trimmed_features)
        assert len(orig_sig_features) == len(trimmed_sig_features)

    @staticmethod
    def validateOutput(output):
        assert output is not None
        assert output.get(DataFormattingService.TRAINING_MATRIX) is not None
        assert output.get(DataFormattingService.TESTING_MATRIX) is not None
        num_train = len(output.get(DataFormattingService.TRAINING_MATRIX).keys())
        num_test = len(output.get(DataFormattingService.TESTING_MATRIX).keys())
        assert num_train > num_test

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
        x_test, x_train, y_test, y_train = self.fetchTrainAndTestData()
        assert (len(x_train) and len(x_test) and len(y_train) and len(y_test) != 0)

    def testStratifySplit(self):
        x_test, x_train, y_test, y_train = self.fetchTrainAndTestData()
        assert (len(x_train) and len(x_test) and len(y_train) and len(y_test) != 0)
        categorical_pd = pd.read_csv(self.current_working_dir +
                                     '/SampleClassifierDataFolder/categorical.csv', delimiter=',')
        data_formatting_service = DataFormattingService(None)
        categorical_onehot = data_formatting_service.oneHot(categorical_pd)
        assert (np.shape(categorical_onehot))[1] == 2

    def testFeatureOrderIsPreserved(self):
        original_input = self.data_formatting_service.inputs.features
        self.data_formatting_service.analyze_all = False  # don't attempt trimming
        formatted_output = self.data_formatting_service.formatData(False, False)
        self.validateMatrixOrderHasNotChanged(formatted_output, original_input, DataFormattingService.TESTING_MATRIX)
        self.validateMatrixOrderHasNotChanged(formatted_output, original_input, DataFormattingService.TRAINING_MATRIX)

    def validateMatrixOrderHasNotChanged(self, formatted_output, original_input, matrix):
        for cell_line in formatted_output.get(matrix).keys():
            formatted_features = formatted_output.get(matrix).get(cell_line)
            original_features = original_input.get(cell_line)
            assert original_features == formatted_features

    def testFeatureScaling(self):
        x_test, x_train, y_test, y_train = self.fetchTrainAndTestData()

        self.scaleFeaturesAndAssert(x_test)
        self.scaleFeaturesAndAssert(x_train)

    def scaleFeaturesAndAssert(self, x_vals):
        feature_one_orig = list(x_vals.get("feature_one"))
        feature_two_orig = list(x_vals.get("feature_two"))
        feature_three_orig = list(x_vals.get("feature_three"))
        scaled_test = self.data_formatting_service.maybeScaleFeatures(x_vals, True)
        assert scaled_test
        scaled_test_vals_as_list = SafeCastUtil.safeCast(scaled_test.values(), list)
        self.assertFeaturesScaled(feature_one_orig, scaled_test_vals_as_list, 0)
        self.assertFeaturesScaled(feature_two_orig, scaled_test_vals_as_list, 1)
        self.assertFeaturesScaled(feature_three_orig, scaled_test_vals_as_list, 2)

    def assertFeaturesScaled(self, feature, scaled_test_vals_as_list, index):
        for i in range(0, len(feature)):
            for j in range(0, len(feature)):
                if feature[i] == feature[j]:
                    assert scaled_test_vals_as_list[i][index] == scaled_test_vals_as_list[j][index]
                elif feature[i] < feature[j]:
                    assert scaled_test_vals_as_list[i][index] < scaled_test_vals_as_list[j][index]
                else:
                    assert scaled_test_vals_as_list[i][index] > scaled_test_vals_as_list[j][index]
