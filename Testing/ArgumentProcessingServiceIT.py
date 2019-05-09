import csv
import unittest
import logging
import os

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms


class ArgumentProcessingServiceIT(unittest.TestCase):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.
        self.total_features_in_files = 150

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                if file == "__init__.py":
                    continue
                os.remove(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testInputTextFileCorrectlyParsed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        self.processAndValidateArguments(input_folder, True)

    def testClassifierWithRandomlyGeneratedInput(self):
        random_data_generator = RandomizedDataGenerator(RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        random_data_generator.generateRandomizedFiles(5, 50, self.total_features_in_files, True, 10, .8)
        assert len(os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)) > 7

        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        self.processAndValidateArguments(input_folder, True)

    def testRegressorWithRandomlyGeneratedInput(self):
        random_data_generator = RandomizedDataGenerator(RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        random_data_generator.generateRandomizedFiles(5, 50, self.total_features_in_files, False, 10, .8)
        assert len(os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER)) > 7

        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        self.processAndValidateArguments(input_folder, False)

    def testFeatureValidation(self):
        random_data_generator = RandomizedDataGenerator(RandomizedDataGenerator.GENERATED_DATA_FOLDER)
        random_data_generator.generateRandomizedFiles(5, 50, self.total_features_in_files, True, 10, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        files = os.listdir(input_folder)
        gene_list = [file for file in files if ArgumentProcessingService.GENE_LISTS in file][0]

        gene_list_line = None
        bogus_gene = "bogus_gene"
        with open(input_folder + "/" + gene_list, 'r') as csv_file:
            try:
                for line_index, line in enumerate(csv_file):
                    gene_list_line = [feature.strip() for feature in line.split(",")]
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", csv_file, error)
            finally:
                csv_file.close()
        gene_list_line.append(bogus_gene)
        with open(input_folder + "/" + gene_list, 'w') as csv_file:
            try:
                writer = csv.writer(csv_file)
                writer.writerow(gene_list_line)
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", csv_file, error)
            finally:
                csv_file.close()
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        assert bogus_gene in arguments.gene_lists.get(gene_list.split(".")[0])

    def testCommentsInArgumentsFileAllowed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        assert arguments.outer_monte_carlo_permutations == 10
        assert arguments.inner_monte_carlo_permutations == 10

    def testArgumentsByAlgorithm(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        rf = SupportedMachineLearningAlgorithms.RANDOM_FOREST
        enet = SupportedMachineLearningAlgorithms.ELASTIC_NET
        assert arguments.algorithm_configs.get(rf) == [True, 5, 5]
        assert arguments.algorithm_configs.get(enet) == [False, 0, 0]

    def testEmptyGeneListsNotProcessed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        gene_lists = SafeCastUtil.safeCast(arguments.gene_lists.keys(), list)
        assert len(gene_lists) == 3
        assert "empty_gene_list" not in gene_lists

    def testSpecificCombosProperlyProcessed(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        combos = arguments.specific_combos
        assert len(combos) == 2
        for combo in combos:
            assert "\"" not in combo

    def processAndValidateArguments(self, input_folder, is_classifier):
        argument_processing_service = ArgumentProcessingService(input_folder)
        arguments = argument_processing_service.handleInputFolder()
        assert arguments is not None
        assert (len(arguments.results) + 1) == len(arguments.features.keys())
        assert arguments.is_classifier == is_classifier
        assert len(arguments.features.get(argument_processing_service.FEATURE_NAMES)) < self.total_features_in_files
        feature_names = arguments.features.get(ArgumentProcessingService.FEATURE_NAMES)
        for cell_line_feature_set in arguments.features.values():
            assert len(cell_line_feature_set) == len(feature_names)
