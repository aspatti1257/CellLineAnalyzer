import unittest
import logging
import os

from MachineLearningService import MachineLearningService
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                os.remove(
                    self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    # def testMachineLearningModelsCreated(self): # Will probably remove non-randomized inputs.
    #     ml_service = MachineLearningService(self.arguments)
    #     self.assertResults(ml_service.analyze(self.current_working_dir))

    def testRandomForestRegressorWithRandomData(self):
        ml_service = MachineLearningService(self.formatRandomizedData(False))
        self.assertResults(ml_service.analyze(self.current_working_dir))

    def testRandomForestClassifierWithRandomData(self):
        ml_service = MachineLearningService(self.formatRandomizedData(True))
        self.assertResults(ml_service.analyze(self.current_working_dir))

    def assertResults(self, rf_results):
        assert rf_results is not None
        for percentage in rf_results.keys():
            assert rf_results[percentage] is not None
            assert type(rf_results[percentage][0]) is float

    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 10, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        return argument_processing_service.handleInputFolder()
