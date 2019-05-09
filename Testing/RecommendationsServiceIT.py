import unittest
import os
import csv

from RecommendationsService import RecommendationsService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil
from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms


class RecommendationsServiceIT(unittest.TestCase):

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        self.DRUG_DIRECTORY = "DrugAnalysisResults"

    def tearDown(self):
        if self.current_working_dir != "/":
            directory = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
            for file_or_dir in os.listdir(directory):
                if file_or_dir == "__init__.py":
                    continue
                current_path = directory + "/" + file_or_dir
                if self.DRUG_DIRECTORY in file_or_dir:
                    for file in os.listdir(current_path):
                        os.remove(current_path + "/" + file)
                    os.removedirs(current_path)
                else:
                    os.remove(current_path)

    # Run this IT and put a breakpoint in recs_service.recommendByHoldout() to see what variables you have to work with.
    def testRecommendations(self):
        inputs = self.formatRandomizedData(False)
        inputs.recs_config.viability_acceptance = 0.1
        self.instantiateMLServiceAndSetupDrugData(inputs)

        try:
            recs_service = RecommendationsService(inputs)
            recs_service.recommendByHoldout(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        except KeyboardInterrupt as keyboard_interrupt:
            assert False

    # These methods below are just setup methods to create a dummy dataset.
    def instantiateMLServiceAndSetupDrugData(self, inputs):
        ml_service = MachineLearningService(inputs)
        combos = [ml_service.generateFeatureSetString(combo) for combo in ml_service.determineGeneListCombos()]
        self.setupDrugData(combos, ml_service)

    def formatRandomizedData(self, is_classifier):
        random_data_generator = RandomizedDataGenerator(RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        random_data_generator.generateRandomizedFiles(3, 1000, 150, is_classifier, 2, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        return argument_processing_service.handleInputFolder()

    def setupDrugData(self, combos, ml_service):
        randomized_data_path = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        for i in range(1, 11):
            drug_name = self.DRUG_DIRECTORY + SafeCastUtil.safeCast(i, str)
            drug_path = randomized_data_path + "/" + drug_name
            if drug_name not in os.listdir(randomized_data_path):
                os.mkdir(drug_path)
            for algo in SupportedMachineLearningAlgorithms.fetchAlgorithms():
                file_name = drug_path + "/" + algo + ".csv"
                with open(file_name, 'w', newline='') as feature_file:
                    writer = csv.writer(feature_file)
                    header = ml_service.getCSVFileHeader(ml_service.inputs.is_classifier,
                                                         algo, ml_service.inputs.outer_monte_carlo_permutations)
                    writer.writerow(header)
                    for combo in combos:
                        row = RandomizedDataGenerator.generateAnalysisRowForCombo(ml_service, combo, algo)
                        writer.writerow(row)
                    feature_file.close()
