import unittest
import logging
import os

from MachineLearningService import MachineLearningService
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Utilities.SafeCastUtil import SafeCastUtil


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                os.remove(
                    self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testRandomForestRegressor(self):
        self.evaluateMachineLearningModel(False, SupportedMachineLearningAlgorithms.RANDOM_FOREST)

    def testRandomForestClassifier(self):
        self.evaluateMachineLearningModel(True, SupportedMachineLearningAlgorithms.RANDOM_FOREST)

    # Linear SVM takes forever to train. Consider a different library?
    def testLinearSVMRegressor(self):
        self.evaluateMachineLearningModel(False, SupportedMachineLearningAlgorithms.LINEAR_SVM)

    def testLinearSVMClassifier(self):
        self.evaluateMachineLearningModel(True, SupportedMachineLearningAlgorithms.LINEAR_SVM)

    def testRadialBasisFunctionSVMRegressor(self):
        self.evaluateMachineLearningModel(False, SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM)

    def testRadialBasisFunctionSVMClassifier(self):
        self.evaluateMachineLearningModel(True, SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM)

    def evaluateMachineLearningModel(self, is_classifier, ml_algorithm):
        ml_service = MachineLearningService(self.formatRandomizedData(is_classifier))
        ml_service.log.setLevel(logging.DEBUG)
        num_gene_list_combos = 4
        gene_list_combos_shortened = ml_service.determineGeneListCombos()[0:num_gene_list_combos]
        monte_carlo_perms = ml_service.inputs.get(ArgumentProcessingService.MONTE_CARLO_PERMUTATIONS)
        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        ml_service.handleParallellization(gene_list_combos_shortened, target_dir, monte_carlo_perms, ml_algorithm)

        self.assertResults(target_dir, ml_algorithm, num_gene_list_combos + 1)

    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 1, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        return argument_processing_service.handleInputFolder()

    def assertResults(self, target_dir, ml_algorithm, expected_lines):
        file_name = ml_algorithm + ".csv"
        assert file_name in os.listdir(target_dir)
        num_lines = 0
        with open(target_dir + "/" + file_name) as csv_file:
            try:
                for line_index, line in enumerate(csv_file):
                    num_lines += 1
                    line_split = line.strip().split(",")
                    if line_index == 0:
                        assert line_split == MachineLearningService.CSV_FILE_HEADER
                        continue
                    feature_gene_list_combo = line_split[0]
                    assert ":" in feature_gene_list_combo
                    score = SafeCastUtil.safeCast(line_split[len(line_split) - 2], float)
                    accuracy = SafeCastUtil.safeCast(line_split[len(line_split) - 1], float)
                    assert score > -1
                    assert accuracy > 0
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                self.log.debug("Closing file %s", file_name)
                csv_file.close()
                assert num_lines == expected_lines

