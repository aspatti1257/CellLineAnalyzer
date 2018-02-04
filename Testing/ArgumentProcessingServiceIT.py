import unittest
import logging
import os

from ArgumentProcessingService import ArgumentProcessingService


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
        assert (len(arguments.get("results")) + 1) == len(arguments.get("features").keys())
        assert arguments.get("is_classifier")

