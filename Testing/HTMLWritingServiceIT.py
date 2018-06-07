import unittest
import os
import logging

from HTMLWritingService import HTMLWritingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService


class HTMLWritingServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                if file == "__init__.py":
                    continue
                os.remove(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testRecordFileWritten(self):
        # self.runMLAnalysis(False)
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        with open(input_folder + "/" + HTMLWritingService.RECORD_FILE) as record_file:
            try:
                for line_index, line in enumerate(record_file):
                    assert line is not None
            except ValueError as value_error:
                self.log.error(value_error)
            finally:
                record_file.close()

        html_service = HTMLWritingService(input_folder)
        html_service.writeSummaryFile()

    def runMLAnalysis(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 10, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        ml_service = MachineLearningService(argument_processing_service.handleInputFolder())
        return ml_service.analyze(input_folder)
