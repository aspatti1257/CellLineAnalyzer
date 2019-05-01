import logging
import os
import unittest
import numpy
import math

from Trainers.ElasticNetTrainer import ElasticNetTrainer
from Trainers.RandomForestTrainer import RandomForestTrainer
from Trainers.LinearSVMTrainer import LinearSVMTrainer
from Trainers.RadialBasisFunctionSVMTrainer import RadialBasisFunctionSVMTrainer
from Trainers.RidgeRegressionTrainer import RidgeRegressionTrainer
from Trainers.LassoRegressionTrainer import LassoRegressionTrainer
from Trainers.RandomSubsetElasticNetTrainer import RandomSubsetElasticNetTrainer

from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Utilities.RandomizedDataGenerator import RandomizedDataGenerator
from Utilities.SafeCastUtil import SafeCastUtil


class MachineLearningServiceIT(unittest.TestCase):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    THRESHOLD_OF_SIGNIFICANCE = 0.60

    MONTE_CARLO_PERMS = 2
    INDIVIDUAL_MONTE_CARLO_PERMS = 10

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

    def testRandomSubsetElasticNet(self):
        ml_service = MachineLearningService(self.formatRandomizedData(False))
        ml_service.log.setLevel(logging.DEBUG)
        binary_cat_matrix = ml_service.inputs.rsen_config.binary_cat_matrix
        rsen_trainer = RandomSubsetElasticNetTrainer(False, binary_cat_matrix, 0, 0.4)

        filtered_combos = self.fetchFilteredRSENCombos(ml_service, rsen_trainer)

        trimmed_combos = filtered_combos[0:8]
        target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        ml_service.handleParallellization(trimmed_combos, target_dir, rsen_trainer)

        self.assertResults(target_dir, rsen_trainer, len(trimmed_combos) + 1, rsen_trainer.is_classifier)

    def fetchFilteredRSENCombos(self, ml_service, rsen_trainer):
        filtered_combos = []
        for combo in ml_service.determineGeneListCombos():
            is_valid = True
            for feature_set in combo:
                if len([feature for feature in feature_set if "bin_cat.significant_feature" in feature]) > 0:
                    is_valid = False
            if is_valid and rsen_trainer.shouldProcessFeatureSet(combo):
                filtered_combos.append(combo)
        return filtered_combos

    def testRandomSubsetElasticNetWithCombinedGeneLists(self):
        inputs = self.formatRandomizedData(False)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        inputs.rsen_config.combine_gene_lists = True
        ml_service = MachineLearningService(inputs)
        ml_service.log.setLevel(logging.DEBUG)
        binary_cat_matrix = ml_service.inputs.rsen_config.binary_cat_matrix
        rsen_trainer = RandomSubsetElasticNetTrainer(False, binary_cat_matrix, 0, 0.4)
        gene_list_combos = ml_service.determineGeneListCombos()

        combos = ml_service.fetchValidGeneListCombos(input_folder, gene_list_combos, rsen_trainer)
        assert len(combos) < len(gene_list_combos)

        for combo in combos:
            assert "ALL_GENE_LISTS" in ml_service.generateFeatureSetString(combo)

    def evaluateMachineLearningModel(self, trainer):
        ml_service = MachineLearningService(self.formatRandomizedData(trainer.is_classifier))
        ml_service.log.setLevel(logging.DEBUG)
        num_gene_list_combos = 8
        self.analyzeAndAssertResults(ml_service, num_gene_list_combos, trainer)

    def analyzeAndAssertResults(self, ml_service, num_gene_list_combos, trainer):
        try:
            gene_list_combos_shortened = ml_service.determineGeneListCombos()[0:num_gene_list_combos]
            target_dir = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
            ml_service.handleParallellization(gene_list_combos_shortened, target_dir, trainer)
            self.assertResults(target_dir, trainer, num_gene_list_combos + 1, trainer.is_classifier)
        except KeyboardInterrupt as keyboardInterrupt:
            self.log.error("Interrupted manually, failing and initiating cleanup.")
            assert False

    def formatRandomizedData(self, is_classifier):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, self.MONTE_CARLO_PERMS, .8)
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        argument_processing_service = ArgumentProcessingService(input_folder)
        argument_processing_service.log.setLevel(logging.DEBUG)
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
                        assert line_split == MachineLearningService.getCSVFileHeader(is_classifier, trainer.algorithm,
                                                                                     self.MONTE_CARLO_PERMS)
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
                    if len(line_split) > 3:
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

    def testIndividualRandomSubsetElasticNet(self):
        self.evaluateMachineLearningModelForIndividualCombo(SupportedMachineLearningAlgorithms.RANDOM_SUBSET_ELASTIC_NET,
                                                            "0.1,0.1", False)

    def evaluateMachineLearningModelForIndividualCombo(self, algorithm, hyperparams, is_classifier):
        input_folder = self.current_working_dir + "/" + RandomizedDataGenerator.GENERATED_DATA_FOLDER
        ml_service = MachineLearningService(self.formatRandomizedDataForIndividualCombo(is_classifier, algorithm,
                                                                                        hyperparams, input_folder))
        if algorithm is SupportedMachineLearningAlgorithms.RANDOM_SUBSET_ELASTIC_NET:
            binary_categorical_matrix = ml_service.inputs.rsen_config.binary_cat_matrix
            dummy_trainer = RandomSubsetElasticNetTrainer(False, binary_categorical_matrix, 0, 0.4)
            target_combo = self.fetchFilteredRSENCombos(ml_service, dummy_trainer)[0]
            target_combo_string = ml_service.generateFeatureSetString(target_combo)
            ml_service.inputs.individual_train_config.combo = target_combo_string

        ml_service.analyze(input_folder)
        self.assertResultsForIndividualCombo(input_folder, algorithm, 11, is_classifier)

    def formatRandomizedDataForIndividualCombo(self, is_classifier, algorithm, hyperparams, input_folder):
        RandomizedDataGenerator.generateRandomizedFiles(3, 1000, 150, is_classifier, self.INDIVIDUAL_MONTE_CARLO_PERMS,
                                                        .8, algorithm, hyperparams)
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
                        assert line_split == MachineLearningService.getCSVFileHeader(is_classifier, algorithm, 1)
                        continue
                    feature_gene_list_combo = line_split[0]
                    assert ":" in feature_gene_list_combo
                    if len(line_split) > 3:
                        top_importance = line_split[3]
                        assert top_importance is not None
            except AssertionError as error:
                self.log.error(error)
            finally:
                self.log.debug("Closing file %s", file_name)
                csv_file.close()
                assert num_lines == expected_lines

    def testTrimmingExistingFeatures(self):
        input_folder = self.current_working_dir + "/SampleClassifierDataFolder"
        argument_processing_service = ArgumentProcessingService(input_folder)
        inputs = argument_processing_service.handleInputFolder()
        ml_service = MachineLearningService(inputs)
        gene_list_combos = ml_service.determineGeneListCombos()
        trainer = RandomForestTrainer(True)
        trimmed_combos = ml_service.fetchValidGeneListCombos(input_folder, gene_list_combos, trainer)
        assert len(trimmed_combos) == (len(gene_list_combos) - 1)

    def testSortingByFeatureImportances(self):
        delimiter = MachineLearningService.DELIMITER
        ml_service = MachineLearningService(None)
        # All columns add up to 1. Equal number of importances for each feature.
        importances = {
            "geneA": [0.0, 0.1, 0.2, 0.4, 0.0],   # total == 0.7
            "geneB": [1.0, 0.1, 0.2, 0.1, 0.5],   # total == 1.9
            "geneC": [0.0, 0.1, 0.2, 0.1, 0.25],  # total == 0.65
            "geneD": [0.0, 0.1, 0.2, 0.3, 0.25],  # total == 0.85
            "geneE": [0.0, 0.6, 0.2, 0.1, 0.0],   # total == 0.9
        }

        sorted_importances1 = ml_service.averageAndSortImportances(importances, 5)
        assert sorted_importances1[0] == "geneB --- 0.38"
        assert sorted_importances1[1] == "geneE --- 0.18"
        assert sorted_importances1[2] == "geneD --- 0.17"
        assert sorted_importances1[3] == "geneA --- 0.14"
        assert sorted_importances1[4] == "geneC --- 0.13"
        assert numpy.sum([SafeCastUtil.safeCast(imp.split(delimiter)[1], float) for imp in sorted_importances1
                          if imp is not ""]) == 1.0

        sorted_importances2 = ml_service.averageAndSortImportances(importances, 6)
        assert len(sorted_importances1) == len(sorted_importances1)
        for i in range(0, len(sorted_importances2)):
            split1 = sorted_importances1[i].split(delimiter)
            split2 = sorted_importances2[i].split(delimiter)
            assert split1[0] == split2[0]
            if split1 == split2:
                continue
            assert SafeCastUtil.safeCast(split1[1], float) > SafeCastUtil.safeCast(split2[1], float)
        assert numpy.sum([SafeCastUtil.safeCast(imp.split(delimiter)[1], float) for imp in sorted_importances2
                          if imp is not ""]) < 1.0

        # 6 columns. Now all the others are missing one.
        importances["geneF"] = [0, 0, 0, 0, 0, 1.0]  # total == 1.0
        sorted_importances3 = ml_service.averageAndSortImportances(importances, 6)
        assert len([imp for imp in sorted_importances3 if imp != ""]) > len([imp for imp in sorted_importances1 if imp != ""])
        assert math.isclose(
            numpy.sum([SafeCastUtil.safeCast(imp.split(delimiter)[1], float) for imp in sorted_importances3
                       if imp is not ""]), 1.0)

        importances["geneG"] = [0, 0, 0, 0, 0, 0, 2.0]  # total == 2.0
        sorted_importances4 = ml_service.averageAndSortImportances(importances, 7)
        assert len([imp for imp in sorted_importances4 if imp != ""]) > len([imp for imp in sorted_importances3 if imp != ""])
        assert numpy.sum([SafeCastUtil.safeCast(imp.split(delimiter)[1], float) for imp in sorted_importances4
                          if imp is not ""]) > 1.0

    def testSpecifiedCombosAreSelectedProperly(self):
        arguments = self.formatRandomizedData(False)
        file_names = []
        for feature in arguments.features.get(ArgumentProcessingService.FEATURE_NAMES):
            file_name = feature.split(".")[0]
            if file_name not in file_names:
                file_names.append(file_name)

        gene_lists = SafeCastUtil.safeCast(arguments.gene_lists.keys(), list)

        self.assertSpecificComboGeneration(arguments, self.generateSpecificCombos(file_names, gene_lists, False))
        self.assertSpecificComboGeneration(arguments, self.generateSpecificCombos(file_names, gene_lists, True))

    def generateSpecificCombos(self, file_names, gene_lists, flip_order):
        specific_combos = []
        if len(file_names) > 1 and len(gene_lists) > 1:
            if flip_order:
                specific_combos.append(file_names[0] + ":" + gene_lists[1] + " " + file_names[1] + ":" + gene_lists[1])
            else:
                specific_combos.append(file_names[1] + ":" + gene_lists[1] + " " + file_names[0] + ":" + gene_lists[1])

        for file in file_names:
            for gene_list in gene_lists:
                if gene_list is not "null_gene_list":
                    specific_combos.append(file + ":" + gene_list)
                    if len(specific_combos) > 4:
                        return specific_combos
        return specific_combos

    def assertSpecificComboGeneration(self, arguments, specific_combos):
        arguments.specific_combos = specific_combos
        ml_service = MachineLearningService(arguments)
        gene_list_combos = ml_service.determineGeneListCombos()
        filtered_combos = ml_service.determineSpecificCombos(gene_list_combos)
        assert len(filtered_combos) == len(specific_combos)

    def testFullAnalysisSansGeneListRandomForestRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(RandomForestTrainer(False))

    def testFullAnalysisSansGeneListRandomForestClassifier(self):
        self.evaluateModelFullAnalysisSansGeneList(RandomForestTrainer(True))

    def testFullAnalysisSansGeneListLinearSVMRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(LinearSVMTrainer(False))

    def testFullAnalysisSansGeneListLinearSVMClassifier(self):
        self.evaluateModelFullAnalysisSansGeneList(LinearSVMTrainer(True))

    def testFullAnalysisSansGeneListRadialBasisFunctionSVMRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(RadialBasisFunctionSVMTrainer(False))

    def testFullAnalysisSansGeneListRadialBasisFunctionSVMClassifier(self):
        self.evaluateModelFullAnalysisSansGeneList(RadialBasisFunctionSVMTrainer(True))

    def testFullAnalysisSansGeneListElasticNetRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(ElasticNetTrainer(False))

    def testFullAnalysisSansGeneListRidgeRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(RidgeRegressionTrainer(False))

    def testFullAnalysisSansGeneListLassoRegressor(self):
        self.evaluateModelFullAnalysisSansGeneList(LassoRegressionTrainer(False))

    def evaluateModelFullAnalysisSansGeneList(self, trainer):
        processed_args = self.formatRandomizedData(trainer.is_classifier)
        processed_args.analyze_all = True
        ml_service = MachineLearningService(processed_args)

        ml_service.log.setLevel(logging.DEBUG)
        trainer.log.setLevel(logging.DEBUG)

        self.analyzeAndAssertResults(ml_service, 1, trainer)

