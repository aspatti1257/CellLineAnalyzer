import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy
import os
import csv

import multiprocessing
import threading
from joblib import Parallel, delayed

from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Utilities.SafeCastUtil import SafeCastUtil
from sklearn.metrics import r2_score


class MachineLearningService(object):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    DEFAULT_MIN_SCORE = -1

    def __init__(self, data):
        self.inputs = data

    def analyze(self, input_folder):
        gene_list_combos = self.determineGeneListCombos()
        monte_carlo_perms = self.inputs.get(ArgumentProcessingService.MONTE_CARLO_PERMUTATIONS)

        if not self.inputs.get(ArgumentProcessingService.SKIP_RF):
            num_models_to_create = monte_carlo_perms * 2 * len(gene_list_combos) * (6 * 4)
            self.log.info("Running permutations on %s different combinations of features. Requires creation of %s "
                          "different Random Forest models.", SafeCastUtil.safeCast(len(gene_list_combos), str),
                          SafeCastUtil.safeCast(num_models_to_create, str))
            self.handleParallellization(gene_list_combos, input_folder, monte_carlo_perms,
                                        SupportedMachineLearningAlgorithms.RANDOM_FOREST)

        if not self.inputs.get(ArgumentProcessingService.SKIP_SVM):
            num_models_to_create = monte_carlo_perms * 2 * len(gene_list_combos) * (3 * 7)
            if not self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
                num_models_to_create = num_models_to_create * 5
            self.log.info("Running permutations on %s different combinations of features. Requires creation of %s "
                          "different Support Vector Machine models.", SafeCastUtil.safeCast(len(gene_list_combos), str),
                          SafeCastUtil.safeCast(num_models_to_create, str))
            self.handleParallellization(gene_list_combos, input_folder, monte_carlo_perms,
                                        SupportedMachineLearningAlgorithms.LINEAR_SVM)

        return

    def handleParallellization(self, gene_list_combos, input_folder, monte_carlo_perms, ml_algorithm):
        num_nodes = multiprocessing.cpu_count()

        Parallel(n_jobs=num_nodes)(delayed(self.runMonteCarloSelection)(feature_set,

                                                                        monte_carlo_perms, ml_algorithm, input_folder)
                                   for feature_set in gene_list_combos)

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
        required_permutations = num_gene_lists**num_files
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
        return list(numpy.zeros(length, dtype=numpy.int))

    def runMonteCarloSelection(self, feature_set, monte_carlo_perms, ml_algorithm, input_folder):
        accuracies = []
        feature_set_as_string = self.generateFeatureSetString(feature_set)
        for i in range(1, monte_carlo_perms + 1):
            formatted_data = self.formatData(self.inputs)
            training_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set, formatted_data)
            testing_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set, formatted_data)

            self.log.info("Computing outer Monte Carlo Permutation %s for %s.", i, feature_set_as_string)

            optimal_hyperparams = self.determineOptimalHyperparameters(feature_set, formatted_data, monte_carlo_perms,
                                                                       ml_algorithm)
            features, results = self.populateFeaturesAndResultsByCellLine(training_matrix)
            accuracies.append(self.fetchOuterPermutationModelScore(feature_set_as_string, features, ml_algorithm,
                                                                   optimal_hyperparams, results, testing_matrix,
                                                                   training_matrix))

        average_accuracy = numpy.mean(accuracies)
        self.log.info("Total accuracy of all Monte Carlo runs for %s: %s", feature_set_as_string, average_accuracy)
        self.writeToCSVInLock(average_accuracy, feature_set_as_string, input_folder, ml_algorithm)

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
                        feature_set_string += (file_key + ":"+ gene_list_key + " ")
        return feature_set_string.strip()

    def fetchOuterPermutationModelScore(self, feature_set_as_string, features, ml_algorithm, optimal_hyperparams,
                                        results, testing_matrix, training_matrix):
        if ml_algorithm == SupportedMachineLearningAlgorithms.RANDOM_FOREST:
            n = len(SafeCastUtil.safeCast(training_matrix.keys(), list))  # number of samples
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "m_val = %s\n" +
                          "max depth = %s", feature_set_as_string, ml_algorithm, optimal_hyperparams[0],
                          optimal_hyperparams[1] * n)
            model = self.trainRandomForest(results, features, optimal_hyperparams[0], optimal_hyperparams[1] * n)
        elif ml_algorithm == SupportedMachineLearningAlgorithms.LINEAR_SVM:
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "c_val = %s\n" +
                          "gamma = %s\n" +
                          "epsilon = %s\n", feature_set_as_string, ml_algorithm, optimal_hyperparams[0],
                          optimal_hyperparams[1], optimal_hyperparams[2])
            model = self.trainLinearSVM(results, features, optimal_hyperparams[0], optimal_hyperparams[1],
                                        optimal_hyperparams[2])
        else:
            return self.DEFAULT_MIN_SCORE
        model_score = self.predictModelAccuracy(model, testing_matrix)
        return model_score

    def determineInnerHyperparameters(self, feature_set, formatted_data, monte_carlo_perms, ml_algorithm):
        inner_model_hyperparams = {}
        for j in range(1, monte_carlo_perms + 1):
            formatted_inputs = self.reformatInputsByTrainingMatrix(
                formatted_data.get(DataFormattingService.TRAINING_MATRIX))
            further_formatted_data = self.formatData(formatted_inputs)
            inner_validation_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TESTING_MATRIX, feature_set,
                                                                  further_formatted_data)
            inner_train_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set,
                                                             further_formatted_data)
            if ml_algorithm == SupportedMachineLearningAlgorithms.RANDOM_FOREST:
                model_data = self.hyperparameterizeForRF(inner_train_matrix, inner_validation_matrix)
            elif ml_algorithm == SupportedMachineLearningAlgorithms.LINEAR_SVM:
                model_data = self.hyperparameterizeForSVM(inner_train_matrix, inner_validation_matrix)
            else:
                return inner_model_hyperparams
            for data in model_data.keys():
                if inner_model_hyperparams.get(data) is not None:
                    inner_model_hyperparams[data].append(model_data[data])
                else:
                    inner_model_hyperparams[data] = [model_data[data]]
            return inner_model_hyperparams

    def formatData(self, inputs):
        data_formatting_service = DataFormattingService(inputs)
        return data_formatting_service.formatData()

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

    def determineOptimalHyperparameters(self, feature_set, formatted_data, monte_carlo_perms, ml_algorithm):
        inner_model_hyperparams = self.determineInnerHyperparameters(feature_set, formatted_data, monte_carlo_perms,
                                                                     ml_algorithm)
        highest_average = self.DEFAULT_MIN_SCORE
        best_hyperparam = None
        for hyperparam_set in inner_model_hyperparams.keys():
            average = numpy.average(inner_model_hyperparams[hyperparam_set])
            if average > highest_average:
                best_hyperparam = hyperparam_set
                highest_average = average
        return best_hyperparam

    def trimMatrixByFeatureSet(self, matrix_type, gene_lists, formatted_inputs):
        full_matrix = formatted_inputs.get(matrix_type)
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURES).get(ArgumentProcessingService.FEATURE_NAMES)
        important_indices = []
        for i in range(0, len(feature_names)):
            for gene_list in gene_lists:
                for gene in gene_list:
                    if gene == feature_names[i]:
                        important_indices.append(i)

        trimmed_matrix = {}
        for cell_line in full_matrix.keys():
            new_cell_line_features = []
            for j in range(0, len(full_matrix[cell_line])):
                if j in important_indices:
                    new_cell_line_features.append(full_matrix[cell_line][j])
            trimmed_matrix[cell_line] = new_cell_line_features
        return trimmed_matrix

    def hyperparameterizeForRF(self, training_matrix, validation_matrix):
        model_data = {}
        p = len(SafeCastUtil.safeCast(training_matrix.values(), list))  # number of features
        n = len(SafeCastUtil.safeCast(training_matrix.keys(), list))  # number of samples
        features, results = self.populateFeaturesAndResultsByCellLine(training_matrix)
        for m_val in [1, (1 + numpy.sqrt(p)) / 2, numpy.sqrt(p), (numpy.sqrt(p) + p) / 2, p]:
            for max_depth in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]:
                model = self.trainRandomForest(results, features, m_val, max_depth * n)
                current_model_score = self.predictModelAccuracy(model, validation_matrix)
                model_data[m_val, max_depth] = current_model_score
        return model_data

    def hyperparameterizeForSVM(self, training_matrix, validation_matrix):
        model_data = {}
        features, results = self.populateFeaturesAndResultsByCellLine(training_matrix)
        for c_val in [10E-2, 10E-1, 10E0]:  # 10E1, 10E2, 10E3, 10E4, 10E5, 10E6, take way too long to train.
            for gamma in [10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 10E0, 10E1]:
                if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
                    model = self.trainLinearSVM(results, features, c_val, gamma, None)
                    current_model_score = self.predictModelAccuracy(model, validation_matrix)
                    model_data[c_val, gamma, None] = current_model_score
                else:
                    for epsilon in [0.01, 0.05, 0.1, 0.15, 0.2]:
                        model = self.trainLinearSVM(results, features, c_val, gamma, epsilon)
                        current_model_score = self.predictModelAccuracy(model, validation_matrix)
                        model_data[c_val, gamma, epsilon] = current_model_score
        return model_data

    def populateFeaturesAndResultsByCellLine(self, matrix):
        features = []
        results = []
        for cell in matrix.keys():
            features.append(matrix[cell])
            for result in self.inputs.get(ArgumentProcessingService.RESULTS):
                if result[0] == cell:
                    results.append(result[1])
        return features, results

    def trainRandomForest(self, results, features, m_val, max_depth):
        max_leaf_nodes = numpy.maximum(2, SafeCastUtil.safeCast(numpy.ceil(max_depth), int))
        max_features = numpy.min([SafeCastUtil.safeCast(numpy.floor(m_val), int), len(features[0])])
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_features=max_features)
        else:
            model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_features=max_features)
        model.fit(features, results)
        self.log.debug("Successful creation of Random Forest model: %s\n", model)
        return model

    def trainLinearSVM(self, results, features, c_val, gamma, epsilon):
        if epsilon is None or self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            model = svm.SVC(kernel='linear', C=c_val, gamma=gamma)
        else:
            model = svm.SVR(kernel='linear', C=c_val, gamma=gamma, epsilon=epsilon)
        model.fit(features, results)
        self.log.debug("Successful creation of Support Vector Machine model: %s\n", model)
        return model

    def predictModelAccuracy(self, model, validation_matrix):
        if model is None:
            return self.DEFAULT_MIN_SCORE
        features, results = self.populateFeaturesAndResultsByCellLine(validation_matrix)
        predictions = model.predict(features)
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            accurate_hits = 0
            for i in range(0, len(predictions)):
                if predictions[i] == results[i]:
                    accurate_hits += 1
            return accurate_hits / len(predictions)
        else:
            return SafeCastUtil.safeCast(r2_score(results, predictions), float)

    def writeToCSVInLock(self, average_accuracy, feature_set_as_string, input_folder, ml_algorithm):
        lock = threading.Lock()
        self.log.debug("Locking current thread %s.", threading.current_thread())
        lock.acquire(True)

        file_name = ml_algorithm + ".csv"
        write_action = "w"
        if file_name in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + file_name, write_action) as csv_file:
            try:
                writer = csv.writer(csv_file)
                writer.writerow([feature_set_as_string, average_accuracy])
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", file_name, error)
            finally:
                csv_file.close()
                os.chdir(input_folder)
                self.log.debug("Releasing current thread %s.", threading.current_thread())
                lock.release()

    def analysisType(self):
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            return "classifier"
        else:
            return "regressor"
