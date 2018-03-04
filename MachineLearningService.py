import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy

from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from Utilities.SafeCastUtil import SafeCastUtil
from sklearn.metrics import r2_score


class MachineLearningService(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    TRAINING_PERCENTS = [20, 40, 60, 80, 100]  # Percentage of training data to actually train on.
    NUM_PERMUTATIONS = 1  # Create and train optimized ML models to get a range of accuracies.

    def __init__(self, data):
        self.inputs = data

    def analyze(self):
        self.log.info(" Initializing Random Forest training with the following features:\n %s",
                      self.inputs.get(ArgumentProcessingService.FEATURE_NAMES))
        total_accuracies = {}
        training_matrix = self.inputs.get(DataFormattingService.TRAINING_MATRIX)
        validation_matrix = self.inputs.get(DataFormattingService.VALIDATION_MATRIX)
        testing_matrix = self.inputs.get(DataFormattingService.TESTING_MATRIX)
        for percent in self.TRAINING_PERCENTS:
            accuracies = []
            split_train_training_matrix = self.furtherSplitTrainingMatrix(percent, training_matrix)
            for permutation in range(0, self.NUM_PERMUTATIONS):
                most_accurate_model = self.optimizeHyperparametersForRF(split_train_training_matrix, validation_matrix)
                accuracy = self.predictModelAccuracy(most_accurate_model, testing_matrix)
                accuracies.append(accuracy)
                self.log.debug("Random Forest Model trained with accuracy: %s", accuracy)
            total_accuracies[percent] = accuracies

        self.log.info(" Total accuracies by percentage of training data for %s: %s", self.analysisType(),
                      total_accuracies)
        return total_accuracies

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
