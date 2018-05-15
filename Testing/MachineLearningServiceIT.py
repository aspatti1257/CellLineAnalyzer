import logging
import os
import unittest

from Trainers.ElasticNetTrainer import ElasticNetTrainer
from Trainers.RandomForestTrainer import RandomForestTrainer
from Trainers.LinearSVMTrainer import LinearSVMTrainer
from Trainers.RadialBasisFunctionSVMTrainer import RadialBasisFunctionSVMTrainer
from Trainers.LinearRegressionTrainer import LinearRegressionTrainer

from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    THRESHOLD_OF_SIGNIFICANCE = 0.60

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        if self.current_working_dir != "/":
            for file in os.listdir(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER):
                if file == "__init__.py":
                    continue
                os.remove(self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER + "/" + file)

    def testRandomForestRegressor(self):
        self.evaluateMachineLearningModel(RandomForestTrainer(False))

    def testRandomForestClassifier(self):
        self.evaluateMachineLearningModel(RandomForestTrainer(True))

    def testLinearSVMRegressor(self):
        self.evaluateMachineLearningModel(LinearSVMTrainer(False))

    def testLinearSVMClassifier(self):
        self.evaluateMachineLearningModel(LinearSVMTrainer(True))

    def testRadialBasisFunctionSVMRegressor(self):
        self.evaluateMachineLearningModel(RadialBasisFunctionSVMTrainer(False))

    def testRadialBasisFunctionSVMClassifier(self):
        self.evaluateMachineLearningModel(RadialBasisFunctionSVMTrainer(True))

    def testElasticNetRegressor(self):
        self.evaluateMachineLearningModel(ElasticNetTrainer(False))

    def testLinearRegressor(self):
        self.evaluateMachineLearningModel(LinearRegressionTrainer(False))

    def evaluateMachineLearningModel(self, trainer):
        ml_service = MachineLearningService(self.formatRandomizedData(trainer.is_classifier))
        ml_service.log.setLevel(logging.DEBUG)
        num_gene_list_combos = 8
        gene_list_combos_shortened = ml_service.determineGeneListCombos()[0:num_gene_list_combos]
        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        ml_service.handleParallellization(gene_list_combos_shortened, target_dir, trainer)

        self.assertResults(target_dir, trainer, num_gene_list_combos + 1, trainer.is_classifier)

    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 1, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        return argument_processing_service.handleInputFolder()

    def assertResults(self, target_dir, trainer, expected_lines, is_classifier):
        file_name = trainer.algorithm + ".csv"
        assert file_name in os.listdir(target_dir)
        num_lines = 0
        with open(target_dir + "/" + file_name) as csv_file:
            try:
                for line_index, line in enumerate(csv_file):
                    num_lines += 1
                    line_split = line.strip().split(",")
                    if line_index == 0:
                        assert line_split == MachineLearningService.getCSVFileHeader(is_classifier)
                        continue
                    feature_gene_list_combo = line_split[0]
                    assert ":" in feature_gene_list_combo
                    score = SafeCastUtil.safeCast(line_split[len(line_split) - 2], float)
                    accuracy = SafeCastUtil.safeCast(line_split[len(line_split) - 1], float)
                    assert score > trainer.DEFAULT_MIN_SCORE
                    if RandomizedDataGenerator.SIGNIFICANT_GENE_LIST in feature_gene_list_combo:
                        assert score >= self.THRESHOLD_OF_SIGNIFICANCE
                    else:
                        assert score < self.THRESHOLD_OF_SIGNIFICANCE
                    assert accuracy > 0
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                self.log.debug("Closing file %s", file_name)
                csv_file.close()
                assert num_lines == expected_lines
