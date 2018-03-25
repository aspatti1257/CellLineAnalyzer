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
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                os.remove(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testInputTextFileCorrectlyParsed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        self.processAndValidateArguments(input_folder, True)

    def testClassifierWithRandomlyGeneratedInput(self):
        RandomizedDataGenerator.generateRandomizedFiles(5, 50, 150, True, 10, .8)
        assert len(os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)) > 7

        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        self.processAndValidateArguments(input_folder, True)

    def testRegressorWithRandomlyGeneratedInput(self):
        RandomizedDataGenerator.generateRandomizedFiles(5, 50, 150, False, 10, .8)
        assert len(os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)) > 7

        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        self.processAndValidateArguments(input_folder, False)

    def processAndValidateArguments(self, input_folder, is_classifier):
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        assert arguments is not None
        assert len(arguments) == 6
        assert (len(arguments.get(argument_processing_service.RESULTS)) + 1) == \
                len(arguments.get(argument_processing_service.FEATURES).keys())
        assert arguments.get(argument_processing_service.IS_CLASSIFIER) == is_classifier
