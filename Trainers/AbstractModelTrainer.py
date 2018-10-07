import logging
import threading

import numpy
import os

from abc import ABC, abstractmethod

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class AbstractModelTrainer(ABC):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    DEFAULT_MIN_SCORE = -10

    ADDITIONAL_DATA = "additional_data"

    @abstractmethod
    def __init__(self, algorithm, hyperparameters, is_classifier):
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.is_classifier = is_classifier

    @abstractmethod
    def hyperparameterize(self, training_matrix, testing_matrix, results):
        pass

    @abstractmethod
    def train(self, results, features, hyperparams, feature_names):
        pass

    @abstractmethod
    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        pass

    @abstractmethod
    def logOptimalHyperParams(self, hyperparams, feature_set_as_string, record_diagnostics, input_folder):
        pass

    @abstractmethod
    def supportsHyperparams(self):
        pass

    @abstractmethod
    def fetchFeatureImportances(self, model, gene_list_combo):
        pass

    def preserveNonHyperparamData(self, model_data, model):
        pass

    def shouldProcessFeatureSet(self, feature_set):
        return True

    def fetchModelPhrases(self, model, gene_list_combo):
        return {}

    def logTrainingMessage(self, outer_monte_carlo_perms, inner_monte_carlo_perms, num_gene_list_combos):
        num_models = self.determineNumModelsToCreate(outer_monte_carlo_perms, inner_monte_carlo_perms, num_gene_list_combos)
        self.log.info("Running permutations on %s different combinations of features. Requires creation of %s "
                      "different %s models.", SafeCastUtil.safeCast(num_gene_list_combos, str),
                      num_models, self.algorithm)

    def determineNumModelsToCreate(self, outer_monte_carlo_perms, inner_monte_carlo_perms, num_gene_list_combos):
        num_models = outer_monte_carlo_perms * inner_monte_carlo_perms * num_gene_list_combos
        for hyperparam_set in self.hyperparameters.values():
            num_models *= len(hyperparam_set)
        return num_models

    def loopThroughHyperparams(self, hyperparams, training_matrix, testing_matrix, results):
        self.hyperparameters = hyperparams

        features, relevant_results = self.populateFeaturesAndResultsByCellLine(training_matrix, results)
        feature_names = training_matrix.get(ArgumentProcessingService.FEATURE_NAMES)

        model_data = {}
        for hyperparam_set in self.fetchAllHyperparamPermutations(hyperparams):
            model = self.train(relevant_results, features, hyperparam_set, feature_names)
            self.preserveNonHyperparamData(model_data, model)
            current_model_score = self.fetchPredictionsAndScore(model, testing_matrix, results)
            self.setModelDataDictionary(model_data, hyperparam_set, current_model_score)
        return model_data

    def fetchAllHyperparamPermutations(self, hyperparams):
        all_perms = []
        hyperparam_keys = SafeCastUtil.safeCast(hyperparams.keys(), list)
        zero_filled_indices = SafeCastUtil.safeCast(numpy.zeros(len(hyperparam_keys)), list)
        target_index = len(zero_filled_indices) - 1
        current_perm = zero_filled_indices[:]
        while target_index >= 0:
            current_hyperparams = []
            for i in range(0, len(current_perm)):
                current_hyperparams.append(hyperparams[hyperparam_keys[i]][SafeCastUtil.safeCast(current_perm[i], int)])
            if current_hyperparams not in all_perms:
                clone_array = current_hyperparams[:]
                all_perms.append(clone_array)

            if current_perm[target_index] < len(hyperparams[hyperparam_keys[target_index]]) - 1:
                current_perm[target_index] += 1
                while len(current_perm) > target_index + 1 and current_perm[target_index + 1] <\
                        len(hyperparams[hyperparam_keys[target_index]]):
                    target_index += 1
            else:
                target_index -= 1
                for subsequent_index in range(target_index, len(current_perm) - 1):
                    current_perm[subsequent_index + 1] = 0
        return all_perms

    def fetchPredictionsAndScore(self, model, testing_matrix, results):
        if model is None:
            return self.DEFAULT_MIN_SCORE
        features, relevant_results = self.populateFeaturesAndResultsByCellLine(testing_matrix, results)
        predictions = model.predict(features)
        score = AbstractModelTrainer.DEFAULT_MIN_SCORE
        try:
            score = model.score(features, relevant_results)
        except ValueError as valueError:
            self.log.error(valueError)
            model.score(features, relevant_results)
        if self.is_classifier:
            accuracy = accuracy_score(relevant_results, predictions)
        else:
            accuracy = mean_squared_error(relevant_results, predictions)
        del model
        return score, accuracy

    def populateFeaturesAndResultsByCellLine(self, matrix, results):
        features = []
        relevant_results = []
        for cell in matrix.keys():
            if cell == ArgumentProcessingService.FEATURE_NAMES:
                continue
            features.append(matrix[cell])
            for result in results:
                if result[0] == cell:
                    relevant_results.append(result[1])
        return features, relevant_results

    def logIfBestHyperparamsOnRangeThreshold(self, best_hyperparams, record_diagnostics, input_folder):
        if not self.supportsHyperparams():
            return
        hyperparam_keys = SafeCastUtil.safeCast(self.hyperparameters.keys(), list)
        for i in range(0, len(hyperparam_keys)):
            hyperparam_set = self.hyperparameters[hyperparam_keys[i]]
            if best_hyperparams[i] >= hyperparam_set[len(hyperparam_set) - 1]:
                message = "Best hyperparam for " + self.algorithm + " on upper threshold of provided hyperparam " \
                          "set: " + hyperparam_keys[i] + " = " + SafeCastUtil.safeCast(best_hyperparams[i], str) + "\n"
                self.log.debug(message)
                if record_diagnostics:
                    self.writeToDiagnosticsFile(input_folder, message)
            elif best_hyperparams[i] <= hyperparam_set[0]:
                message = "Best hyperparam for " + self.algorithm + " on lower threshold of provided hyperparam " \
                          "set: " + hyperparam_keys[i] + " = " + SafeCastUtil.safeCast(best_hyperparams[i], str) + "\n"
                self.log.debug(message)
                if record_diagnostics:
                    self.writeToDiagnosticsFile(input_folder, message)

    def writeToDiagnosticsFile(self, input_folder, message):
        lock = threading.Lock()
        lock.acquire(True)

        write_action = "w"
        file_name = "Diagnostics.txt"
        if file_name in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + file_name, write_action) as diagnostics_file:
            try:
                diagnostics_file.write(message)
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", diagnostics_file, error)
            finally:
                diagnostics_file.close()
                # TODO: Lock thread message here too. Extract to logging utility or something.
                lock.release()

    def generateFeaturesInOrder(self, gene_list_combo):
        features_in_order = []
        for feature_file in gene_list_combo:
            for feature in feature_file:
                features_in_order.append(feature)
        return features_in_order

    def normalizeCoefficients(self, coefficients, features_in_order):
        importances = {}
        absolute_sum = numpy.sum([numpy.abs(coeff) for coeff in coefficients])
        for i in range(0, len(features_in_order)):
            if absolute_sum > 0:
                importances[features_in_order[i]] = numpy.abs(coefficients[i]) / absolute_sum
            else:
                importances[features_in_order[i]] = numpy.abs(coefficients[i])  # should be 0.
        return importances
