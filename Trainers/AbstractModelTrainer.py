import threading

import numpy
import copy
import os

from collections import OrderedDict
from abc import ABC, abstractmethod

from ArgumentProcessingService import ArgumentProcessingService
from LoggerFactory import LoggerFactory
from Utilities.DiagnosticsFileWriter import DiagnosticsFileWriter
from Utilities.DictionaryUtility import DictionaryUtility
from Utilities.SafeCastUtil import SafeCastUtil
from Utilities.GarbageCollectionUtility import GarbageCollectionUtility
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class AbstractModelTrainer(ABC):

    log = LoggerFactory.createLog(__name__)

    DEFAULT_MIN_SCORE = -10

    ADDITIONAL_DATA = "additional_data"

    EMPTY_MODEL_RESPONSE = DEFAULT_MIN_SCORE, 0.0

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
    def supportsHyperparams(self):
        pass

    @abstractmethod
    def fetchFeatureImportances(self, model, features_in_order):
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
        return num_models + (outer_monte_carlo_perms * num_gene_list_combos)

    def loopThroughHyperparams(self, hyperparams, training_matrix, testing_matrix, results):
        self.hyperparameters = hyperparams

        features, relevant_results = self.populateFeaturesAndResultsByCellLine(training_matrix, results)
        feature_names = training_matrix.get(ArgumentProcessingService.FEATURE_NAMES)

        hyperparam_permutations = self.fetchAllHyperparamPermutations(hyperparams)
        GarbageCollectionUtility.logMemoryUsageAndGarbageCollect(self.log)
        return self.hyperparameterizeInSerial(feature_names, features, hyperparam_permutations,
                                              relevant_results, results, testing_matrix)

    def hyperparameterizeInSerial(self, feature_names, features, hyperparam_permutations, relevant_results,
                                  results, testing_matrix):
        model_data = {}
        for hyperparam_set in hyperparam_permutations:
            self.buildModelAndRecordScore(feature_names, features, hyperparam_set, model_data, relevant_results,
                                          results, testing_matrix)
        return model_data

    def chunkList(self, original_list, size):
        return [original_list[i * size:(i + 1) * size] for i in range((len(original_list) + size - 1) // size)]

    def buildModelAndRecordScore(self, feature_names, features, hyperparam_set, model_data, relevant_results, results,
                                 testing_matrix):
        if not isinstance(model_data, dict) and model_data._address_to_local is not None:
            shared_file = SafeCastUtil.safeCast(model_data._address_to_local.keys(), list)[0]
            if not os.path.exists(shared_file):
                self.log.warning("Unable to find shared file %s, process likely ended prematurely.", shared_file)
                return

        self.log.debug("Building %s model with hyperparams %s.", self.algorithm, hyperparam_set)
        model = self.buildModel(relevant_results, features, hyperparam_set, feature_names)
        self.preserveNonHyperparamData(model_data, model)
        current_model_score = self.fetchPredictionsAndScore(model, testing_matrix, results)

        lock = threading.Lock()
        lock.acquire(True)
        try:
            model_data[DictionaryUtility.toString(hyperparam_set)] = current_model_score
        except FileNotFoundError as fnfe:
            self.log.error("Unable to write to shared model_data object for algorithm: %s.\n", fnfe)
        except AttributeError as ae:
            self.log.error("Unable to write to shared model_data object for algorithm: %s.\n", ae)
        finally:
            lock.release()

        self.log.debug("Finished building %s model with hyperparams %s.", self.algorithm, hyperparam_set)
        return model_data

    def buildModel(self, relevant_results, features, hyperparam_set, feature_names):
        model = None
        try:
            model = self.train(relevant_results, features, hyperparam_set, feature_names)
        except ValueError as valueError:
            self.log.error("Failed to create model build for %s:\n%s", self.algorithm, valueError)
        return model

    def fetchAllHyperparamPermutations(self, hyperparams):
        all_perms = []
        hyperparam_keys = SafeCastUtil.safeCast(hyperparams.keys(), list)
        zero_filled_indices = SafeCastUtil.safeCast(numpy.zeros(len(hyperparam_keys)), list)
        target_index = len(zero_filled_indices) - 1
        current_perm = zero_filled_indices[:]
        while target_index >= 0:
            current_hyperparams = OrderedDict()
            for i in range(0, len(current_perm)):
                param_name = hyperparam_keys[i]
                current_hyperparams[param_name] = hyperparams[param_name][SafeCastUtil.safeCast(current_perm[i], int)]
            if current_hyperparams not in all_perms:
                clone_map = copy.deepcopy(current_hyperparams)
                all_perms.append(clone_map)

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
            return self.EMPTY_MODEL_RESPONSE
        features, relevant_results = self.populateFeaturesAndResultsByCellLine(testing_matrix, results)
        predictions = model.predict(features)
        score = AbstractModelTrainer.DEFAULT_MIN_SCORE
        try:
            score = model.score(features, relevant_results)
        except ValueError as valueError:
            self.log.error(valueError)
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
        if not self.supportsHyperparams() or best_hyperparams is None:
            return
        hyperparam_keys = SafeCastUtil.safeCast(self.hyperparameters.keys(), list)
        for i in range(0, len(hyperparam_keys)):
            hyperparam_set = self.hyperparameters[hyperparam_keys[i]]
            optimal_value = best_hyperparams.get(hyperparam_keys[i])
            if optimal_value is None:
                self.log.warn("Unable to determine optimal value given hyperparams: %s",
                              SafeCastUtil.safeCast(best_hyperparams, str, None))
                continue
            if optimal_value >= hyperparam_set[len(hyperparam_set) - 1]:
                message = "Best hyperparam for " + self.algorithm + " on upper threshold of provided hyperparam " \
                          "set: " + hyperparam_keys[i] + " = " + SafeCastUtil.safeCast(optimal_value, str) + "\n"
                self.log.debug(message)
                if record_diagnostics:
                    DiagnosticsFileWriter.writeToFile(input_folder, message, self.log)
            elif optimal_value <= hyperparam_set[0]:
                message = "Best hyperparam for " + self.algorithm + " on lower threshold of provided hyperparam " \
                          "set: " + hyperparam_keys[i] + " = " + SafeCastUtil.safeCast(optimal_value, str) + "\n"
                self.log.debug(message)
                if record_diagnostics:
                    DiagnosticsFileWriter.writeToFile(input_folder, message, self.log)

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string, record_diagnostics, input_folder):
        message = "Optimal Hyperparameters for " + feature_set_as_string + " " + self.algorithm + " algorithm " \
                  "chosen as:\n"

        for key in SafeCastUtil.safeCast(hyperparams.keys(), list):
            message += "\t" + key + " = " + SafeCastUtil.safeCast(hyperparams[key], str) + "\n"
        self.log.info(message)
        if record_diagnostics:
            DiagnosticsFileWriter.writeToFile(input_folder, message, self.log)

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
