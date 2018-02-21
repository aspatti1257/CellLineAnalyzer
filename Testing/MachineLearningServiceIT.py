import unittest
import logging
import os

from MachineLearningService import MachineLearningService
from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        current_working_dir = os.getcwd()  # Should be this package.
        input_folder = current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        data_formatting_service = DataFormattingService(arguments)
        self.arguments = data_formatting_service.formatData()

    def testMachineLearningModelsCreated(self):
        ml_service = MachineLearningService(self.arguments)
        rf_results = ml_service.analyze()
        assert rf_results is not None
        for percentage in rf_results.keys():
            assert rf_results[percentage] is not None
            assert type(rf_results[percentage][0]) is float
