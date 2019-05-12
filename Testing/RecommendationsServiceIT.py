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

        try:
            recs_service = RecommendationsService(inputs)
            recs_service.recommendByHoldout(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        except KeyboardInterrupt as keyboard_interrupt:
            assert False


    def formatRandomizedData(self, is_classifier):
        randomized_data_path = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        processed_arguments = {}
        for i in range(1, 11):
            drug_name = self.DRUG_DIRECTORY + SafeCastUtil.safeCast(i, str)
            drug_path = randomized_data_path + "/" + drug_name
            if drug_name not in os.listdir(randomized_data_path):
                os.mkdir(drug_path)
            random_data_generator = RandomizedDataGenerator(drug_path)
            random_data_generator.generateRandomizedFiles(3, 1000, 150, is_classifier, 2, .8)
            argument_processing_service = ArgumentProcessingService(drug_path)
            processed_args = argument_processing_service.handleInputFolder()
            processed_args.recs_config.viability_acceptance = 0.1
            processed_arguments[drug_name] = processed_args
            ml_service = MachineLearningService(processed_args)
            combos = [ml_service.generateFeatureSetString(combo) for combo in ml_service.determineGeneListCombos()]
            self.setupDrugData(combos, ml_service, drug_path)

        return processed_arguments

    def setupDrugData(self, combos, ml_service, drug_path):
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
