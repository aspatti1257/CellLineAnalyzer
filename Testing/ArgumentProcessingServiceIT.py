import unittest
import logging
import os

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator


class ArgumentProcessingServiceIT(unittest.TestCase):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        pass

    def testInputTextFileCorrectlyParsed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        assert arguments is not None
        assert len(arguments) == 3
        assert (len(arguments.get(argument_processing_service.RESULTS)) + 1) == \
               len(arguments.get(argument_processing_service.FEATURES).keys())
        assert arguments.get(argument_processing_service.IS_CLASSIFIER)

    def testWithRandomlyGeneratedInput(self):
        #TODO: finish this test and add more like it for the other ITs.
        RandomizedDataGenerator.generateCSVs(5, 50, 150, True)
