import logging
import threading

import numpy
import os

from abc import ABC, abstractmethod

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import multiprocessing


class AbstractModelTrainer(ABC):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    DEFAULT_MIN_SCORE = -10

    ADDITIONAL_DATA = "additional_data"

    EMPTY_MODEL_RESPONSE = 0.0, 0.0

    parallel_hyperparam_threads = -1

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
        return num_models

    def loopThroughHyperparams(self, hyperparams, training_matrix, testing_matrix, results):
        self.hyperparameters = hyperparams

        features, relevant_results = self.populateFeaturesAndResultsByCellLine(training_matrix, results)
        feature_names = training_matrix.get(ArgumentProcessingService.FEATURE_NAMES)


        hyperparam_permutations = self.fetchAllHyperparamPermutations(hyperparams)
        if SafeCastUtil.safeCast(self.parallel_hyperparam_threads, int, -1) < 0:
            return self.hyperparameterizeInSerial(feature_names, features, hyperparam_permutations,
                                                  relevant_results, results, testing_matrix)
        else:
            return self.hyperparameterizeInParallel(feature_names, features, hyperparam_permutations,
                                                    relevant_results, results, testing_matrix)

    def hyperparameterizeInSerial(self, feature_names, features, hyperparam_permutations, relevant_results,
                                  results, testing_matrix):
        model_data = {}
        for hyperparam_set in hyperparam_permutations:
            self.buildModelAndRecordScore(feature_names, features, hyperparam_set, model_data, relevant_results,
                                          results, testing_matrix)
        return model_data

    def hyperparameterizeInParallel(self, feature_names, features, hyperparam_permutations,
                                    relevant_results, results, testing_matrix):
        model_data = {}
        chunked_hyperparams = self.chunkList(hyperparam_permutations, self.parallel_hyperparam_threads)
        for chunk in chunked_hyperparams:
            manager = multiprocessing.Manager()
            parallelized_dict = manager.dict()
            multithreaded_jobs = []
            for hyperparam_set in chunk:
                process = multiprocessing.Process(target=self.buildModelAndRecordScore,
                                                  args=(feature_names, features, hyperparam_set, parallelized_dict,
                                                        relevant_results, results, testing_matrix))
                multithreaded_jobs.append(process)

                try:
                    process.start()
                except:
                    self.log.error("Multithreaded job failed during individual hyperparam analysis for algorithm %s.",
                                   self.algorithm)

            for proc in multithreaded_jobs:
                try:
                    proc.join()
                except:
                    self.log.error("Multithreaded job failed during individual hyperparam analysis for algorithm %s "
                                   "and PID %s.", self.algorithm, proc.pid)

            for hyperparam_set in chunk:
                copied_hyperparams = hyperparam_set[:]
                if len(copied_hyperparams) == 1:
                    copied_hyperparams.append(None)
                hyperparam_tuple = SafeCastUtil.safeCast(copied_hyperparams, tuple)
                optimal_value = parallelized_dict.get(hyperparam_tuple)
                if optimal_value is not None:
                    model_data[hyperparam_tuple] = optimal_value
                else:
                    self.log.warning("Parallel hyperparameter optimization thread for %s did not execute successfully. "
                                     "No training data available for hyperparams: %s.", self.algorithm, hyperparam_set)
                    model_data[hyperparam_tuple] = self.EMPTY_MODEL_RESPONSE
            additional_data = parallelized_dict.get(self.ADDITIONAL_DATA)
            if additional_data is not None:
                if model_data.get(self.ADDITIONAL_DATA) is None:
                    model_data[self.ADDITIONAL_DATA] = additional_data
                else:
                    for add_data in additional_data:
                        model_data[self.ADDITIONAL_DATA].append(add_data)

        return model_data

    def chunkList(self, original_list, size):
        return [original_list[i * size:(i + 1) * size] for i in range((len(original_list) + size - 1) // size)]

    def buildModelAndRecordScore(self, feature_names, features, hyperparam_set, model_data, relevant_results, results,
                                 testing_matrix):
        self.log.debug("Building %s model with hyperparams %s.", self.algorithm, hyperparam_set)
        model = self.buildModel(relevant_results, features, hyperparam_set, feature_names)
        self.preserveNonHyperparamData(model_data, model)
        current_model_score = self.fetchPredictionsAndScore(model, testing_matrix, results)

        lock = threading.Lock()
        lock.acquire(True)
        try:
            self.setModelDataDictionary(model_data, hyperparam_set, current_model_score)
        except FileNotFoundError as fnfe:
            self.log.error("Unable to write to shared model_date object for algorithm: %s.\n", fnfe)
        except AttributeError as ae:
            self.log.error("Unable to write to shared model_date object for algorithm: %s.\n", ae)
        finally:
            lock.release()

        self.log.debug("Finished building %s model with hyperparams %s.", self.algorithm, hyperparam_set)
        return model_data

    def buildModel(self, relevant_results, features, hyperparam_set, feature_names):
        model = None
        try:
            model = self.train(relevant_results, features, hyperparam_set, feature_names)
        except ValueError as valueError:
            self.log.error("Failed to create model build for %s: \n%s", self.algorithm, valueError)
        return model

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
