import unittest
import logging
import os

from MachineLearningService import MachineLearningService
from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        data_formatting_service = DataFormattingService(arguments)
        self.arguments = data_formatting_service.formatData()

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                os.remove(
                    self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testMachineLearningModelsCreated(self):
        ml_service = MachineLearningService(self.arguments)
        self.assertResults(ml_service.analyze())

    def testRandomForestRegressorWithRandomData(self):
        ml_service = MachineLearningService(self.formatRandomizedData(False))
        self.assertResults(ml_service.analyze())

    def testRandomForestClassifierWithRandomData(self):
        ml_service = MachineLearningService(self.formatRandomizedData(True))
        self.assertResults(ml_service.analyze())

    def assertResults(self, rf_results):
        assert rf_results is not None
        for percentage in rf_results.keys():
            assert rf_results[percentage] is not None
            assert type(rf_results[percentage][0]) is float

    # TODO: DRY this up. Repeated form DataFormattingServiceIT. Maybe make a method in utility.
    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(5, 50, 150, is_classifier)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        data_formatting_service = DataFormattingService(arguments)
        return data_formatting_service.formatData()

