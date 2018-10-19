import unittest
import os
import csv

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
            dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
            for file_or_dir in os.listdir(dir):
                if file_or_dir == "__init__.py":
                    continue
                current_path = dir + "/" + file_or_dir
                if self.DRUG_DIRECTORY in file_or_dir:
                    for file in os.listdir(current_path):
                        os.remove(current_path + "/" + file)
                    os.removedirs(current_path)
                else:
                    os.remove(current_path)

    def testRecommendations(self):
        ml_service = MachineLearningService(self.formatRandomizedData(False))
        combos = [ml_service.generateFeatureSetString(combo) for combo in ml_service.determineGeneListCombos()]
        self.setupDrugData(combos, ml_service)
        pass

    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 2, .8)
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
                    header = ml_service.getCSVFileHeader(ml_service.inputs.get(ArgumentProcessingService.IS_CLASSIFIER),
                                                         algo, ml_service.inputs.get(ArgumentProcessingService.OUTER_MONTE_CARLO_PERMUTATIONS))
                    writer.writerow(header)
                    for combo in combos:
                        row = RandomizedDataGenerator.generateAnalysisRowForCombo(ml_service, combo, algo)
                        writer.writerow(row)
                    feature_file.close()
