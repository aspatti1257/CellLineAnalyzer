import unittest
import logging
import os

from MachineLearningService import MachineLearningService
from ArgumentProcessingService import ArgumentProcessingService


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        current_working_dir = os.getcwd()  # Should be this package.
        input_folder = current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        self.arguments = argument_processing_service.handleInputFolder()

    def testMachineLearningModelsCreated(self):
        ml_service = MachineLearningService(self.arguments)
        basic_rf_model = ml_service.analyze()
        assert basic_rf_model is not None
        assert len(basic_rf_model.feature_importances_) == 4
