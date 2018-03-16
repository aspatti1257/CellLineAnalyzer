import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy
import os
import csv

from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from Utilities.SafeCastUtil import SafeCastUtil
from sklearn.metrics import r2_score


class MachineLearningService(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    TRAINING_PERCENTS = [20, 40, 60, 80, 100]  # Percentage of training data to actually train on.
    NUM_PERMUTATIONS = 1  # Create and train optimized ML models to get a range of accuracies. Currently unused.

    def __init__(self, data):
        self.inputs = data

    def analyze(self, input_folder):
        accuracies_by_gene_set = {}
        feature_set_combos = self.recursivelyDetermineFeatureSetCombos()

        self.log.info("Running permutations on %s different combinations of features", len(feature_set_combos))

        file_name = "RandomForestAnalysis.csv"
        with open(file_name, 'w') as csv_file:
            try:
                writer = csv.writer(csv_file)
                for feature_set in feature_set_combos:
                    training_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX, feature_set)
                    validation_matrix = self.trimMatrixByFeatureSet(DataFormattingService.VALIDATION_MATRIX,
                                                                    feature_set)
                    testing_matrix = self.trimMatrixByFeatureSet(DataFormattingService.TESTING_MATRIX, feature_set)
                    feature_set_as_string = SafeCastUtil.safeCast(feature_set, str)
                    self.log.info("Training Random Forest with feature set: %s", feature_set_as_string)
                    accuracies = {}
                    for percent in self.TRAINING_PERCENTS:
                        split_train_training_matrix = self.furtherSplitTrainingMatrix(percent, training_matrix)
                        most_accurate_model = self.optimizeHyperparametersForRF(split_train_training_matrix,
                                                                                validation_matrix)
                        accuracy = self.predictModelAccuracy(most_accurate_model, testing_matrix)
                        accuracies[percent] = accuracy
                        self.log.debug("Random Forest Model trained with accuracy: %s", accuracy)
                    self.log.info("Accuracies by percent for %s: %s", feature_set_as_string, accuracies)
                    accuracies_by_gene_set[feature_set_as_string] = accuracies
                    writer.writerow([feature_set_as_string, accuracies])

            except ValueError as error:
                self.log.error("Error writing to file %s. %s", file_name, error)
            finally:
                csv_file.close()
                os.chdir(input_folder)

        self.log.info(" Total accuracies by percentage of training data for %s: %s", self.analysisType(),
                      accuracies_by_gene_set)
        return accuracies_by_gene_set

    def recursivelyDetermineFeatureSetCombos(self):
        #TODO: Fails if parsing up to 5^5 (3125) separate combos.
        gene_sets_across_files = {}
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURE_NAMES)
        for feature in feature_names:
            split = feature.split(".")
            if gene_sets_across_files.get(split[0]) is not None:
                gene_sets_across_files[split[0]].append(feature)
            else:
                gene_sets_across_files[split[0]] = [feature]
        current_combo = []
        feature_set_combos = []
        parsed_feature_files = []
        feature_set_combos = self.generateFeatureSetCombos(gene_sets_across_files, current_combo,
                                                           feature_set_combos, parsed_feature_files)
        return feature_set_combos

    def generateFeatureSetCombos(self, gene_sets_across_files, current_combo,
                                 feature_set_combos, parsed_feature_files):
        num_permutations = len(list(gene_sets_across_files.values())[0])**len(list(gene_sets_across_files.keys()))
        if len(feature_set_combos) >= num_permutations:
            return feature_set_combos

        for feature_file in gene_sets_across_files.keys():
            if feature_file not in parsed_feature_files:
                all_features_in_this_file_in_current_combo = True
                for feature in gene_sets_across_files[feature_file]:
                    if feature not in current_combo:
                        all_features_in_this_file_in_current_combo = False
                        parsed_feature_files.append(feature_file)
                        current_combo.append(feature)
                        break

                if all_features_in_this_file_in_current_combo:
                    for feature in gene_sets_across_files[feature_file]:
                        current_combo.remove(feature)
                    if len(parsed_feature_files) > 0:
                        parsed_feature_files.pop(len(parsed_feature_files) - 1)
                    self.generateFeatureSetCombos(gene_sets_across_files, current_combo, feature_set_combos,
                                                  parsed_feature_files)
                    return feature_set_combos

        cloned_combo = current_combo[:]
        feature_set_combos.append(cloned_combo)
        if len(parsed_feature_files) > 0:
            parsed_feature_files.pop(len(parsed_feature_files) - 1)
            self.generateFeatureSetCombos(gene_sets_across_files, current_combo, feature_set_combos,
                                          parsed_feature_files)
        return feature_set_combos

    def trimMatrixByFeatureSet(self, matrix_type, feature_set):
        full_matrix = self.inputs.get(matrix_type)
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURE_NAMES)
        important_indices = []
        for i in range(0, len(feature_names)):
            for feature in feature_set:
                if feature == feature_names[i]:
                    important_indices.append(i)

        trimmed_matrix = {}
        for cell_line in full_matrix.keys():
            new_cell_line_features = []
            for j in range(0, len(full_matrix[cell_line])):
                if j in important_indices:
                    new_cell_line_features.append(full_matrix[cell_line][j])
            trimmed_matrix[cell_line] = new_cell_line_features
        return trimmed_matrix

    def furtherSplitTrainingMatrix(self, percent, matrix):
        self.log.debug("Splitting training matrix to only use %s percent of data.", percent)
        new_matrix_len = SafeCastUtil.safeCast(len(matrix.keys()) * (percent / 100), int)
        split_matrix = {}
        for cell_line in SafeCastUtil.safeCast(matrix.keys(), list):
            if len(split_matrix.keys()) < new_matrix_len:
                split_matrix[cell_line] = matrix[cell_line]
        return matrix

    def optimizeHyperparametersForRF(self, training_matrix, validation_matrix):
        most_accurate_model = None
        most_accurate_model_score = -1
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            most_accurate_model_score = 0
        p = len(SafeCastUtil.safeCast(training_matrix.values(), list))  # number of features
        n = len(SafeCastUtil.safeCast(training_matrix.keys(), list))  # number of samples
        for m_val in [1, (1 + numpy.sqrt(p)) / 2, numpy.sqrt(p), (numpy.sqrt(p) + p) / 2, p]:
            for max_depth in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]:
                features, results = self.populateFeaturesAndResultsByCellLine(training_matrix)
                model = self.trainRandomForest(results, features, m_val, max_depth*n)
                current_model_score = self.predictModelAccuracy(model, validation_matrix)
                if current_model_score > most_accurate_model_score:
                    most_accurate_model_score = current_model_score
                    most_accurate_model = model
        return most_accurate_model

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

    def predictModelAccuracy(self, model, validation_matrix):
        if model is None:
            return 0
        features, results = self.populateFeaturesAndResultsByCellLine(validation_matrix)
        predictions = model.predict(features)
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            accurate_hits = 0
            for i in range(0, len(predictions)):
                if predictions[i] == results[i]:
                    accurate_hits += 1
            return accurate_hits/len(predictions)
        else:
            return SafeCastUtil.safeCast(r2_score(results, predictions), float)

    def analysisType(self):
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            return "classifier"
        else:
            return "regressor"
