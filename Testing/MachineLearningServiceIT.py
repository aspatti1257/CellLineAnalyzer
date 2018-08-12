import logging
import os
import unittest

from Trainers.ElasticNetTrainer import ElasticNetTrainer
from Trainers.RandomForestTrainer import RandomForestTrainer
from Trainers.LinearSVMTrainer import LinearSVMTrainer
from Trainers.RadialBasisFunctionSVMTrainer import RadialBasisFunctionSVMTrainer
from Trainers.RidgeRegressionTrainer import RidgeRegressionTrainer
from Trainers.LassoRegressionTrainer import LassoRegressionTrainer
from Trainers.RandomSubsetLinearRegressionTrainer import RandomSubsetLinearRegressionTrainer

from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
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

    def testRidgeRegressor(self):
        self.evaluateMachineLearningModel(RidgeRegressionTrainer(False))

    def testLassoRegressor(self):
        self.evaluateMachineLearningModel(LassoRegressionTrainer(False))


    def testRandomSubsetLinearRegressor(self):  # TODO: DRY this up and get it passing!
        ml_service = MachineLearningService(self.formatRandomizedData(False))
        ml_service.log.setLevel(logging.DEBUG)
        binary_cat_matrix = ml_service.inputs.get(ArgumentProcessingService.BINARY_CATEGORICAL_MATRIX)
        rslr_trainer = RandomSubsetLinearRegressionTrainer(False, binary_cat_matrix)

        num_gene_list_combos = 8
        gene_list_combos_shortened = ml_service.determineGeneListCombos()[0:num_gene_list_combos]
        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        # ml_service.handleParallellization(gene_list_combos_shortened, target_dir, rslr_trainer)

        # self.assertResults(target_dir, rslr_trainer, num_gene_list_combos + 1, rslr_trainer.is_classifier)

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
        self.assertDiagnosticResults(target_dir, trainer)

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
                    score = SafeCastUtil.safeCast(line_split[1], float)
                    accuracy = SafeCastUtil.safeCast(line_split[2], float)
                    assert score > trainer.DEFAULT_MIN_SCORE
                    if RandomizedDataGenerator.SIGNIFICANT_GENE_LIST in feature_gene_list_combo:
                        assert score >= self.THRESHOLD_OF_SIGNIFICANCE
                    else:
                        assert score < self.THRESHOLD_OF_SIGNIFICANCE
                    assert accuracy > 0
                    top_importance = line_split[3]
                    assert top_importance is not None
            except AssertionError as error:
                self.log.error(error)
            finally:
                self.log.debug("Closing file %s", file_name)
                csv_file.close()
                assert num_lines == expected_lines

    def assertDiagnosticResults(self, target_dir, trainer):
        if trainer.supportsHyperparams():
            diagnostics_file = "Diagnostics.txt"
            if diagnostics_file in os.listdir(target_dir):
                with open(target_dir + "/" + diagnostics_file) as open_file:
                    try:
                        for line_index, line in enumerate(open_file):
                            if "Best Hyperparam" in line:
                                assert trainer.algorithm in line
                                assert "upper" in line or "lower" in line
                    except ValueError as valueError:
                        self.log.error(valueError)
                    finally:
                        self.log.debug("Closing file %s", open_file)
                        open_file.close()

    def testIndividualRandomForestRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RANDOM_FOREST,
                                                            "400,2", False)

    def testIndividualRandomForestClassifier(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RANDOM_FOREST,
                                                            "400,2", True)

    def testIndividualLinearSVMRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.LINEAR_SVM, "0.1,0,1",
                                                            False)

    def testIndividualLinearSVMClassifier(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.LINEAR_SVM, "0.1",
                                                            True)

    def testIndividualRadialBasisFunctionSVMRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM,
                                                            "0.1,0.1,0.1", False)

    def testIndividualRadialBasisFunctionSVMClassifier(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM,
                                                            "0.1,0.1,0.1", True)

    def testIndividualElasticNetRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.ELASTIC_NET, "0.1,0.1",
                                                            False)

    def testIndividualRidgeRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RIDGE_REGRESSION,
                                                            "1", False)

    def testIndividualLassoRegressor(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.LASSO_REGRESSION,
                                                            "1", False)

    def testIndividualRandomPartitionLinearRegressor(self):
        # TODO:
        # self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RIDGE_REGRESSION,
        #  None, False)
        assert True

    def evaluateMachineLearningModelForIndividualCombo(self, algorithm, hyperparams, is_classifier):
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        ml_service = MachineLearningService(self.formatRandomizedDataForIndividualCombo(is_classifier, algorithm,
                                                                                        hyperparams, input_folder))
        ml_service.analyze(input_folder)
        self.assertResultsForIndividualCombo(input_folder, algorithm, 11, is_classifier)

    def formatRandomizedDataForIndividualCombo(self, is_classifier, algorithm, hyperparams, input_folder):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, 10, .8, algorithm, hyperparams)
        argument_processing_service = ArgumentProcessingService(input_folder)
        return argument_processing_service.handleInputFolder()

    def assertResultsForIndividualCombo(self, target_dir, algorithm, expected_lines, is_classifier):
        file_name = algorithm + ".csv"
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
                    top_importance = line_split[3]
                    assert top_importance is not None
            except AssertionError as error:
                self.log.error(error)
            finally:
                self.log.debug("Closing file %s", file_name)
                csv_file.close()
                assert num_lines == expected_lines
