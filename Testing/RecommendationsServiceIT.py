import unittest
import os
import csv
import string
import random

from LoggerFactory import LoggerFactory
from RecommendationsService import RecommendationsService
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil
from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms


class RecommendationsServiceIT(unittest.TestCase):

    log = LoggerFactory.createLog(__name__)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        self.DRUG_DIRECTORY = "DrugAnalysisResults"
        self.NUM_DRUGS = 10

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

    def testRecommendations(self):
        num_cell_lines = 30
        inputs = self.formatRandomizedData(False, num_cell_lines)
        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER

        try:
            recs_service = RecommendationsService(inputs)
            recs_service.recommendByHoldout(target_dir)

            file_name = target_dir + "/" + RecommendationsService.PREDICTIONS_FILE
            num_lines = 0
            drug_names = SafeCastUtil.safeCast(recs_service.inputs.keys(), list)

            with open(file_name) as txt_file:
                try:
                    for line_index, line in enumerate(txt_file):
                        num_lines += 1
                        line_split = line.split(",")

                        if line_index == 0:
                            assert line_split[0] == "Drug"
                        else:
                            assert line_split[0] in drug_names
                            assert "cell_line" in line_split[1]
                            assert SafeCastUtil.safeCast(line_split[2], float) is not None
                            assert SafeCastUtil.safeCast(line_split[3].strip(), float) is not None
                except AssertionError as error:
                    self.log.error(error)
                finally:
                    self.log.debug("Closing file %s", file_name)
                    txt_file.close()
                    assert num_lines == (num_cell_lines * self.NUM_DRUGS) + 1

        except KeyboardInterrupt as keyboard_interrupt:
            assert False

    def testPreRecsAnalysis(self):
        num_cell_lines = 1000
        inputs = self.formatRandomizedData(False, num_cell_lines)
        for processed_arguments in inputs.values():
            sample_features = processed_arguments.features.get(RandomizedDataGenerator.CELL_LINE + "0")
            for _ in range(10):
                num_cell_lines += 1
                self.addRandomCellLine(processed_arguments, sample_features)

        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER

        try:
            recs_service = RecommendationsService(inputs)
            recs_service.preRecsAnalysis(target_dir)

            file_name = target_dir + "/" + RecommendationsService.PRE_REC_ANALYSIS_FILE
            num_lines = 0
            drug_names = SafeCastUtil.safeCast(recs_service.inputs.keys(), list)
            cell_line = RandomizedDataGenerator.CELL_LINE
            with open(file_name) as csv_file:
                try:
                    for line_index, line in enumerate(csv_file):
                        num_lines += 1
                        line_split = line.split(",")
                        for i in range(0, len(line_split)):
                            value_in_csv = line_split[i].strip()
                            if line_index == 0:
                                if i == 0:
                                    assert value_in_csv == cell_line
                                else:
                                    assert value_in_csv == drug_names[i - 1]
                            else:
                                if i == 0:
                                    assert cell_line or RecommendationsService.STD_DEVIATION or \
                                           RecommendationsService.MEAN or RecommendationsService.MEDIAN in value_in_csv
                                else:
                                    assert value_in_csv == MachineLearningService.DELIMITER.strip() or \
                                           SafeCastUtil.safeCast(value_in_csv, float) > AbstractModelTrainer.DEFAULT_MIN_SCORE

                except AssertionError as error:
                    self.log.error(error)
                finally:
                    self.log.debug("Closing file %s", file_name)
                    csv_file.close()
                    assert num_lines == num_cell_lines + 4
        except KeyboardInterrupt as keyboard_interrupt:
            assert False

    def addRandomCellLine(self, processed_arguments, sample_features):
        random_string = self.randomString(16)
        processed_arguments.features[random_string] = sample_features
        processed_arguments.results.append([random_string, random.random()])

    def randomString(self, string_length):
        letters = string.hexdigits
        return ''.join(random.choice(letters) for i in range(string_length))

    def formatRandomizedData(self, is_classifier, num_cell_lines):
        randomized_data_path = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        processed_arguments = {}
        for i in range(self.NUM_DRUGS):
            drug_name = self.DRUG_DIRECTORY + SafeCastUtil.safeCast(i + 1, str)
            drug_path = randomized_data_path + "/" + drug_name
            if drug_name not in os.listdir(randomized_data_path):
                os.mkdir(drug_path)
            random_data_generator = RandomizedDataGenerator(drug_path)
            random_data_generator.generateRandomizedFiles(3, num_cell_lines, 150, is_classifier, 2, .8)
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
