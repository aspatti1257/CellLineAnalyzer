import csv
import gc
import logging
import multiprocessing
import numpy
import os
import threading

from joblib import Parallel, delayed

from HTMLWritingService import HTMLWritingService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from Trainers.RandomForestTrainer import RandomForestTrainer
from Trainers.LinearSVMTrainer import LinearSVMTrainer
from Trainers.RadialBasisFunctionSVMTrainer import RadialBasisFunctionSVMTrainer
from Trainers.ElasticNetTrainer import ElasticNetTrainer
from Trainers.RidgeRegressionTrainer import RidgeRegressionTrainer
from Trainers.LassoRegressionTrainer import LassoRegressionTrainer
from Trainers.RandomSubsetLinearRegressionTrainer import RandomSubsetLinearRegressionTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class MachineLearningService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    MAXIMUM_FEATURES_RECORDED = 20

    def __init__(self, data):
        self.inputs = data

    def analyze(self, input_folder):
        gene_list_combos = self.determineGeneListCombos()
        is_classifier = self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER)

        if self.inputs.get(ArgumentProcessingService.INDIVIDUAL_TRAIN_FEATURE_GENE_LIST_COMBO) is not None\
                and self.inputs.get(ArgumentProcessingService.INDIVIDUAL_TRAIN_ALGORITHM) is not None:
            self.analyzeIndividualGeneListCombo(gene_list_combos, input_folder, is_classifier)
        else:
            self.analyzeAllGeneListCombos(gene_list_combos, input_folder, is_classifier)

    def determineGeneListCombos(self):
        gene_lists = self.inputs.get(ArgumentProcessingService.GENE_LISTS)
        gene_sets_across_files = {}
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURES).get(ArgumentProcessingService.FEATURE_NAMES)
        for feature in feature_names:
            split = feature.split(".")
            if gene_sets_across_files.get(split[0]) is not None:
                gene_sets_across_files[split[0]].append(feature)
            else:
                gene_sets_across_files[split[0]] = [feature]

        numerical_permutations = self.generateNumericalPermutations(gene_lists, gene_sets_across_files)
        gene_list_keys = SafeCastUtil.safeCast(gene_lists.keys(), list)
        file_keys = SafeCastUtil.safeCast(gene_sets_across_files.keys(), list)
        gene_list_combos = []
        for perm in numerical_permutations:
            feature_strings = []
            for i in range(0, len(perm)):
                file_name = file_keys[i]
                gene_list = gene_lists[gene_list_keys[SafeCastUtil.safeCast(perm[i], int)]]
                if len(gene_list) > 0:
                    feature_strings.append([file_name + "." + gene for gene in gene_list])
            if len(feature_strings) > 0:
                gene_list_combos.append(feature_strings)

        return gene_list_combos

    def generateNumericalPermutations(self, gene_lists, gene_sets_across_files):
        num_gene_lists = len(gene_lists)
        num_files = len(gene_sets_across_files)
        all_arrays = self.fetchAllArrayPermutations((num_gene_lists - 1), num_files)
        required_permutations = num_gene_lists ** num_files
        created_permutations = len(all_arrays)
        self.log.debug("Should have created %s permutations, created %s permutations", required_permutations,
                       created_permutations)
        return all_arrays

    def fetchAllArrayPermutations(self, max_depth, num_files):
        all_arrays = []
        current_array = self.blankArray(num_files)
        target_index = num_files - 1
        while target_index >= 0:
            if current_array not in all_arrays:
                clone_array = current_array[:]
                all_arrays.append(clone_array)
            if current_array[target_index] < max_depth:
                current_array[target_index] += 1
                while len(current_array) > target_index + 1 and current_array[target_index + 1] < max_depth:
                    target_index += 1
            else:
                target_index -= 1
                for subsequent_index in range(target_index, len(current_array) - 1):
                    current_array[subsequent_index + 1] = 0
        return all_arrays

    def blankArray(self, length):
        return SafeCastUtil.safeCast(numpy.zeros(length, dtype=numpy.int), list)

    def analyzeIndividualGeneListCombo(self, gene_list_combos, input_folder, is_classifier):
        target_combo = self.inputs.get(ArgumentProcessingService.INDIVIDUAL_TRAIN_FEATURE_GENE_LIST_COMBO)
        target_algorithm = self.inputs.get(ArgumentProcessingService.INDIVIDUAL_TRAIN_ALGORITHM)
        hyperparams = self.inputs.get(ArgumentProcessingService.INDIVIDUAL_TRAIN_HYPERPARAMS).split(",")
        casted_params = [SafeCastUtil.safeCast(param, float) for param in hyperparams]

        outer_monte_carlo_loops = self.inputs.get(ArgumentProcessingService.OUTER_MONTE_CARLO_PERMUTATIONS)
        for gene_list_combo in gene_list_combos:
            plain_text_name = self.generateFeatureSetString(gene_list_combo)
            if plain_text_name == target_combo:
                trainer = self.createTrainerFromTargetAlgorithm(is_classifier, target_algorithm)
                for permutation in range(0, outer_monte_carlo_loops):
                    results = self.inputs.get(ArgumentProcessingService.RESULTS)
                    formatted_data = self.formatData(self.inputs, trainer.algorithm)
                    training_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX,
                                                                  gene_list_combo, formatted_data)
                    testing_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TESTING_MATRIX, gene_list_combo,
                                                                 formatted_data)
                    features, relevant_results = trainer.populateFeaturesAndResultsByCellLine(training_matrix, results)
                    model = trainer.train(relevant_results, features, casted_params)
                    model_score = trainer.fetchPredictionsAndScore(model, testing_matrix, results)
                    score = model_score[0]
                    accuracy = model_score[1]
                    ordered_importances = self.averageAndSortImportances(self.fetchFeatureImportances(model, gene_list_combo))

                    self.log.debug("Final score and accuracy of individual analysis for feature gene combo %s "
                                   "using algorithm %s: %s, %s", target_combo, target_algorithm, score, accuracy)
                    line = numpy.concatenate([[target_combo, score, accuracy], ordered_importances])
                    self.writeToCSVInLock(line, input_folder, target_algorithm, outer_monte_carlo_loops)
                return
        self.log.info("Gene list feature file %s combo not found in current dataset.", target_combo)
        return

    def createTrainerFromTargetAlgorithm(self, is_classifier, target_algorithm):
        if target_algorithm == SupportedMachineLearningAlgorithms.RANDOM_FOREST:
            trainer = RandomForestTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.LINEAR_SVM:
            trainer = LinearSVMTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM:
            trainer = RadialBasisFunctionSVMTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.ELASTIC_NET and not is_classifier:
            trainer = ElasticNetTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.RIDGE_REGRESSION and not is_classifier:
            trainer = RidgeRegressionTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.LASSO_REGRESSION and not is_classifier:
            trainer = LassoRegressionTrainer(is_classifier)
        # elif target_algorithm == SupportedMachineLearningAlgorithms.RANDOM_PARTITION_LINEAR_REGRESSION \
        #         and not is_classifier:
        #     trainer = RandomPartitionLinearRegressionTrainer(is_classifier, False, None)
        else:
            raise ValueError("Unsupported Machine Learning algorithm: %s", target_algorithm)
        return trainer

    def shouldTrainAlgorithm(self, algorithm):
        return self.inputs.get(ArgumentProcessingService.ALGORITHM_CONFIGS).get(algorithm)[0]

    def analyzeAllGeneListCombos(self, gene_list_combos, input_folder, is_classifier):

        rf = SupportedMachineLearningAlgorithms.RANDOM_FOREST
        lin_svm = SupportedMachineLearningAlgorithms.LINEAR_SVM
        rbf = SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM
        enet = SupportedMachineLearningAlgorithms.ELASTIC_NET
        rig_reg = SupportedMachineLearningAlgorithms.RIDGE_REGRESSION
        las_reg = SupportedMachineLearningAlgorithms.LASSO_REGRESSION
        rslr = SupportedMachineLearningAlgorithms.RANDOM_SUBSET_LINEAR_REGRESSION

        if not self.inputs.get(ArgumentProcessingService.SKIP_RF) and self.shouldTrainAlgorithm(rf):
            rf_trainer = RandomForestTrainer(is_classifier)
            rf_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(rf, True),
                                          self.monteCarloPermsByAlgorithm(rf, False), len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, rf_trainer)

        if not self.inputs.get(ArgumentProcessingService.SKIP_LINEAR_SVM) and self.shouldTrainAlgorithm(lin_svm):
            linear_svm_trainer = LinearSVMTrainer(is_classifier)
            linear_svm_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(lin_svm, True),
                                                  self.monteCarloPermsByAlgorithm(lin_svm, False),
                                                  len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, linear_svm_trainer)

        if not self.inputs.get(ArgumentProcessingService.SKIP_RBF_SVM) and self.shouldTrainAlgorithm(rbf):
            rbf_svm_trainer = RadialBasisFunctionSVMTrainer(is_classifier)
            rbf_svm_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(rbf, True),
                                               self.monteCarloPermsByAlgorithm(rbf, False), len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, rbf_svm_trainer)

        if not self.inputs.get(ArgumentProcessingService.SKIP_ELASTIC_NET) and not is_classifier and\
                self.shouldTrainAlgorithm(enet):
            elasticnet_trainer = ElasticNetTrainer(is_classifier)
            elasticnet_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(enet, True),
                                                  self.monteCarloPermsByAlgorithm(enet, False),
                                                  len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, elasticnet_trainer)

        if not self.inputs.get(ArgumentProcessingService.SKIP_RIDGE_REGRESSION) and not is_classifier and \
                self.shouldTrainAlgorithm(rig_reg):
            ridge_regression_trainer = RidgeRegressionTrainer(is_classifier)
            ridge_regression_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(rig_reg, True),
                                                        self.monteCarloPermsByAlgorithm(rig_reg, False),
                                                        len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, ridge_regression_trainer)

        if not self.inputs.get(ArgumentProcessingService.SKIP_LASSO_REGRESSION) and not is_classifier and \
                self.shouldTrainAlgorithm(las_reg):
            lasso_regression_trainer = LassoRegressionTrainer(is_classifier)
            lasso_regression_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(rig_reg, True),
                                                         self.monteCarloPermsByAlgorithm(rig_reg, False),
                                                         len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, lasso_regression_trainer)

        binary_cat_matrix = self.inputs.get(ArgumentProcessingService.BINARY_CATEGORICAL_MATRIX)
        if self.shouldTrainAlgorithm(rslr) and not is_classifier and binary_cat_matrix is not None:
            rslr_trainer = RandomSubsetLinearRegressionTrainer(is_classifier, binary_cat_matrix)
            rslr_trainer.logTrainingMessage(self.monteCarloPermsByAlgorithm(rslr, True),
                                            self.monteCarloPermsByAlgorithm(rslr, False),
                                            len(gene_list_combos))
            self.handleParallellization(gene_list_combos, input_folder, rslr_trainer)

    def monteCarloPermsByAlgorithm(self, algorithm, outer):
        if outer:
            return self.inputs.get(ArgumentProcessingService.ALGORITHM_CONFIGS).get(algorithm)[1]
        else:
            return self.inputs.get(ArgumentProcessingService.ALGORITHM_CONFIGS).get(algorithm)[2]

    def handleParallellization(self, gene_list_combos, input_folder, trainer):
        max_nodes = multiprocessing.cpu_count()
        requested_threads = self.inputs.get(ArgumentProcessingService.NUM_THREADS)
        nodes_to_use = numpy.amin([requested_threads, max_nodes])

        Parallel(n_jobs=nodes_to_use)(delayed(self.runMonteCarloSelection)(feature_set, trainer, input_folder,
                                                                           len(gene_list_combos))
                                      for feature_set in gene_list_combos)

    def runMonteCarloSelection(self, feature_set, trainer, input_folder, num_combos):
        self.setVariablesOnTrainerInSpecialCases(feature_set, trainer)
        scores = []
        accuracies = []
        importances = {}
        feature_set_as_string = self.generateFeatureSetString(feature_set)
        outer_perms = self.monteCarloPermsByAlgorithm(trainer.algorithm, True)
        for i in range(1, outer_perms + 1):
            gc.collect()
            formatted_data = self.formatData(self.inputs, trainer.algorithm)
            training_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set,
                                                          formatted_data)
            testing_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TESTING_MATRIX, feature_set,
                                                         formatted_data)

            self.log.debug("Computing outer Monte Carlo Permutation %s for %s.", i, feature_set_as_string)

            optimal_hyperparams = self.determineOptimalHyperparameters(feature_set, formatted_data, trainer)
            record_diagnostics = self.inputs.get(ArgumentProcessingService.RECORD_DIAGNOSTICS)
            trainer.logIfBestHyperparamsOnRangeThreshold(optimal_hyperparams, record_diagnostics, input_folder)

            prediction_data = self.fetchOuterPermutationModelScore(feature_set, trainer,
                                                                   optimal_hyperparams, testing_matrix,
                                                                   training_matrix)
            scores.append(prediction_data[0])
            accuracies.append(prediction_data[1])
            for importance in prediction_data[2].keys():
                if importances.get(importance) is not None:
                    importances[importance].append(prediction_data[2].get(importance))
                else:
                    importances[importance] = [prediction_data[2].get(importance)]

        average_score = numpy.mean(scores)
        average_accuracy = numpy.mean(accuracies)
        ordered_importances = self.averageAndSortImportances(importances)

        self.log.debug("Average score and accuracy of all Monte Carlo runs for %s: %s, %s",
                       feature_set_as_string, average_score, average_accuracy)
        line = numpy.concatenate([[feature_set_as_string, average_score, average_accuracy], ordered_importances])
        self.writeToCSVInLock(line, input_folder, trainer.algorithm, num_combos)
        self.saveOutputToTxtFile(scores, accuracies, feature_set_as_string, input_folder, trainer.algorithm)

    def setVariablesOnTrainerInSpecialCases(self, feature_set, trainer):
        if trainer.algorithm == SupportedMachineLearningAlgorithms.RANDOM_SUBSET_LINEAR_REGRESSION:
            trainer.current_feature_set = feature_set

    def generateFeatureSetString(self, feature_set):
        feature_map = {}
        for feature_list in feature_set:
            for feature in feature_list:
                file_name = feature.split(".")[0]
                feature_name = feature.split(".")[1:][0]
                if feature_map.get(file_name):
                    feature_map[file_name].append(feature_name)
                else:
                    feature_map[file_name] = [feature_name]
        gene_lists = self.inputs.get(ArgumentProcessingService.GENE_LISTS)

        feature_set_string = ""
        for file_key in feature_map.keys():
            for gene_list_key in gene_lists.keys():
                if len(feature_map[file_key]) == len(gene_lists[gene_list_key]):
                    feature_map[file_key].sort()
                    gene_lists[gene_list_key].sort()
                    same_list = True
                    for i in range(0, len(gene_lists[gene_list_key])):
                        if gene_lists[gene_list_key][i] != feature_map[file_key][i]:
                            same_list = False
                    if same_list:
                        feature_set_string += (file_key + ":" + gene_list_key + " ")
        return feature_set_string.strip()

    def fetchOuterPermutationModelScore(self, feature_set, trainer, optimal_hyperparams,
                                        testing_matrix, training_matrix):
        # TODO: Handle hyperparams with n
        results = self.inputs.get(ArgumentProcessingService.RESULTS)
        features, relevant_results = trainer.populateFeaturesAndResultsByCellLine(training_matrix, results)
        trainer.logOptimalHyperParams(optimal_hyperparams, self.generateFeatureSetString(feature_set))
        model = trainer.train(relevant_results, features, optimal_hyperparams)
        score, accuracy = trainer.fetchPredictionsAndScore(model, testing_matrix, results)
        importances = self.fetchFeatureImportances(model, feature_set)
        return [score, accuracy, importances]

    def fetchFeatureImportances(self, model, feature_set):
        importances = {}
        features_in_order = []
        for feature_file in feature_set:
            for feature in feature_file:
                features_in_order.append(feature)

        if hasattr(model, "feature_importances_") and len(features_in_order) == len(model.feature_importances_):
            for i in range(0, len(features_in_order)):
                importances[features_in_order[i]] = model.feature_importances_[i]  # already normalized.
        elif hasattr(model, "coef_") or hasattr(model, "support_vectors_"):
            coefficients = []
            if hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
                coefficients = model.coef_
            elif hasattr(model, "coef_") and len(features_in_order) == len(model.coef_[0]):
                coefficients = model.coef_[0]
            elif hasattr(model, "support_vectors_") and len(features_in_order) == len(model.support_vectors_[0]):
                coefficients = model.support_vectors_[0]
            if len(coefficients) == 0:
                return importances
            absolute_sum = numpy.sum([numpy.abs(coeff) for coeff in coefficients])
            for i in range(0, len(features_in_order)):
                if absolute_sum > 0:
                    importances[features_in_order[i]] = numpy.abs(coefficients[i]) / absolute_sum
                else:
                    importances[features_in_order[i]] = numpy.abs(coefficients[i])  # should be 0.

        return importances

    def averageAndSortImportances(self, importances):
        ordered_imps = []
        [ordered_imps.append({"feature": k, "importance": numpy.mean(importances[k])}) for k in importances.keys()]
        ordered_imps = sorted(ordered_imps, key=lambda k: k["importance"])
        ordered_imps.reverse()
        ordered_imps = ordered_imps[:self.MAXIMUM_FEATURES_RECORDED]

        return [imp.get("feature") + " --- " + SafeCastUtil.safeCast(imp.get("importance"), str) for imp in ordered_imps]

    def determineInnerHyperparameters(self, feature_set, formatted_data, trainer):
        inner_model_hyperparams = {}
        inner_perms = self.monteCarloPermsByAlgorithm(trainer.algorithm, False)
        for j in range(1, inner_perms + 1):
            formatted_inputs = self.reformatInputsByTrainingMatrix(
                formatted_data.get(DataFormattingService.TRAINING_MATRIX))
            further_formatted_data = self.formatData(formatted_inputs, trainer.algorithm)
            inner_validation_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TESTING_MATRIX, feature_set,
                                                                  further_formatted_data)
            inner_train_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set,
                                                             further_formatted_data)
            results = self.inputs.get(ArgumentProcessingService.RESULTS)
            model_data = trainer.hyperparameterize(inner_train_matrix, inner_validation_matrix, results)
            for data in model_data.keys():
                if inner_model_hyperparams.get(data) is not None:
                    inner_model_hyperparams[data].append(model_data[data])
                else:
                    inner_model_hyperparams[data] = [model_data[data]]
        return inner_model_hyperparams

    def formatData(self, inputs, algorithm):
        data_formatting_service = DataFormattingService(inputs)
        return data_formatting_service.formatData(algorithm != SupportedMachineLearningAlgorithms.RANDOM_SUBSET_LINEAR_REGRESSION)

    def reformatInputsByTrainingMatrix(self, training_matrix):
        new_inputs = {ArgumentProcessingService.FEATURES: {}, ArgumentProcessingService.RESULTS: []}

        new_inputs[ArgumentProcessingService.FEATURES][ArgumentProcessingService.FEATURE_NAMES] = \
            self.inputs[ArgumentProcessingService.FEATURES][ArgumentProcessingService.FEATURE_NAMES]
        for training_cell in training_matrix.keys():
            for input_cell in self.inputs.get(ArgumentProcessingService.FEATURES).keys():
                if training_cell is input_cell:
                    new_inputs[ArgumentProcessingService.FEATURES][training_cell] = \
                        self.inputs.get(ArgumentProcessingService.FEATURES)[training_cell]
                    for result in self.inputs[ArgumentProcessingService.RESULTS]:
                        if result[0] is training_cell:
                            new_inputs[ArgumentProcessingService.RESULTS].append(result)
                            break
                    break
        new_inputs[ArgumentProcessingService.DATA_SPLIT] = self.inputs[ArgumentProcessingService.DATA_SPLIT]
        return new_inputs

    def determineOptimalHyperparameters(self, feature_set, formatted_data, trainer):
        inner_model_hyperparams = self.determineInnerHyperparameters(feature_set, formatted_data, trainer)
        highest_average = trainer.DEFAULT_MIN_SCORE
        best_hyperparam = None
        for hyperparam_set in inner_model_hyperparams.keys():
            average = numpy.average([results[0] for results in inner_model_hyperparams[hyperparam_set]])  # raw score
            if average > highest_average:
                best_hyperparam = hyperparam_set
                highest_average = average
        return best_hyperparam

    def trimMatrixByFeatureSet(self, matrix_type, gene_lists, formatted_inputs):
        full_matrix = formatted_inputs.get(matrix_type)
        trimmed_matrix = {
            ArgumentProcessingService.FEATURE_NAMES: []
        }

        important_indices = []
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURES).get(ArgumentProcessingService.FEATURE_NAMES)
        for i in range(0, len(feature_names)):
            for gene_list in gene_lists:
                for gene in gene_list:
                    if gene == feature_names[i]:
                        important_indices.append(i)
                        trimmed_matrix[ArgumentProcessingService.FEATURE_NAMES].append(gene)

        for cell_line in full_matrix.keys():
            new_cell_line_features = []
            for j in range(0, len(full_matrix[cell_line])):
                if j in important_indices:
                    new_cell_line_features.append(full_matrix[cell_line][j])
            trimmed_matrix[cell_line] = new_cell_line_features
        return trimmed_matrix

    def writeToCSVInLock(self, line, input_folder, ml_algorithm, num_combos):
        lock = threading.Lock()
        lock.acquire(True)
        self.lockThreadMessage()

        file_name = ml_algorithm + ".csv"
        write_action = "w"
        if file_name in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + file_name, write_action, newline='') as csv_file:
            try:
                writer = csv.writer(csv_file)
                if write_action == "w":
                    writer.writerow(self.getCSVFileHeader(self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER)))
                writer.writerow(line)
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", file_name, error)
            finally:
                csv_file.close()

        total_lines = 0
        with open(input_folder + "/" + file_name) as csv_file:
            try:
                reader = csv.reader(csv_file)
                total_lines += (len(SafeCastUtil.safeCast(reader, list)) - 1)
            except ValueError as error:
                self.log.error("Error reading lines from file %s. %s", file_name, error)
            finally:
                csv_file.close()
                self.logPercentDone(total_lines, num_combos, ml_algorithm)

        self.unlockThreadMessage()
        lock.release()


    @staticmethod
    def getCSVFileHeader(is_classifier):
        header = ["feature file: gene list combo"]
        if is_classifier:
            header.append("percentage accurate predictions")
            header.append("accuracy score")
        else:
            header.append("R^2 score")
            header.append("mean squared error")
        for i in range(1, MachineLearningService.MAXIMUM_FEATURES_RECORDED + 1):
            if i == 1:
                suffix = "st"
            elif i == 2:
                suffix = "nd"
            elif i == 3:
                suffix = "rd"
            else:
                suffix = "th"
            header.append(SafeCastUtil.safeCast(i, str) + suffix + " most important feature")
        return header

    def logPercentDone(self, total_lines, num_combos, ml_algorithm):
        percent_done = numpy.round((total_lines / num_combos) * 100, 1)
        percentage_bar = "["
        for i in range(0, 100):
            if i < percent_done:
                percentage_bar += "="
            elif i >= percent_done:
                if i > 0 and (i - 1) < percent_done:
                    percentage_bar += ">"
                else:
                    percentage_bar += " "
        percentage_bar += "]"
        self.log.info("Total progress for %s: %s%% done: %s", ml_algorithm, percent_done, percentage_bar)

    def saveOutputToTxtFile(self, scores, accuracies, feature_set_as_string, input_folder, algorithm):
        lock = threading.Lock()
        self.lockThreadMessage()
        lock.acquire(True)

        file_name = HTMLWritingService.RECORD_FILE
        write_action = "w"
        if file_name in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + file_name, write_action) as output_file:
            try:
                output_file.write(algorithm + " --- " + feature_set_as_string + " --- " +
                                  SafeCastUtil.safeCast(scores, str) + " --- " + SafeCastUtil.safeCast(accuracies, str)
                                  + "\n")
            except ValueError as error:
                self.log.error("Error saving output of %s analysis to memory: %s", algorithm, error)
            finally:
                output_file.close()
                self.unlockThreadMessage()
                lock.release()

    def lockThreadMessage(self):
        self.log.debug("Locking current thread %s.", threading.current_thread())

    def unlockThreadMessage(self):
        self.log.debug("Releasing current thread %s.", threading.current_thread())
