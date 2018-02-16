import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy

from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from Utilities.SafeCastUtil import SafeCastUtil


class MachineLearningService(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    TRAINING_PERCENTS = [20, 40, 60, 80, 100]  # TODO Percentage of training data to actually train on.
    NUM_PERMUTATIONS = 100  # Create and train 100 optimized ML models to get a range of accuracies.

    def __init__(self, data):
        self.inputs = data

    def analyze(self):
        total_accuracies = []
        training_matrix = self.inputs.get(DataFormattingService.TRAINING_MATRIX)
        validation_matrix = self.inputs.get(DataFormattingService.VALIDATION_MATRIX)
        testing_matrix = self.inputs.get(DataFormattingService.TESTING_MATRIX)
        for percent in range(0, len(self.TRAINING_PERCENTS)):
            split_train_training_matrix = self.furtherSplitTrainingMatrix(self.TRAINING_PERCENTS[percent],
                                                                          training_matrix)
            most_accurate_model = self.optimizeHyperparametersForRF(split_train_training_matrix, validation_matrix)
            accuracy = self.predictModelAccuracy(most_accurate_model, testing_matrix)
            total_accuracies.append(accuracy)
            self.log.debug("Random Forest Model trained with accuracy: %s", accuracy)
        return total_accuracies

    def furtherSplitTrainingMatrix(self, percent, matrix):
        self.log.info(percent, matrix)
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
                features, results = self.populateFeaturesAndResultsByCell(training_matrix)
                model = self.trainRandomForest(results, features, m_val, max_depth)
                current_model_score = self.predictModelAccuracy(model, validation_matrix)
                if current_model_score > most_accurate_model_score:
                    most_accurate_model_score = current_model_score
                    most_accurate_model = model
        return most_accurate_model

    def populateFeaturesAndResultsByCell(self, matrix):
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
        features, results = self.populateFeaturesAndResultsByCell(validation_matrix)
        predictions = model.predict(features)
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            accurate_hits = 0
            for i in range(0, len(predictions)):
                if predictions[i] == results[i]:
                    accurate_hits += 1
            return accurate_hits/len(predictions)
        else:
            # TODO, R2 analaysis for Regressor
            pass
        return model

    def hyperparameterTuning (self, dataframe):
        param_lst = {"n_estimators": range(30, 50)}
        num_trials_outer = 2
        num_trials_inner = 2
        r2_rf = []
        for outerMCCV in range(num_trials_outer):
            out_x_train, out_x_test, out_y_train, out_y_test = train_test_split(dataframe, auc, test_size=0.2, random_state=42)
            out_y_train = out_y_train.flatten()
            out_y_test = out_y_test.flatten()
            param_outer, score_outer, param_inner, score_inner = [], [], [], []
            for innerMCCV in range(num_trials_inner):
                in_x_train, in_x_test, in_y_train, in_y_test = train_test_split(out_x_train, out_y_train, test_size=0.2, random_state=42)
                clf = RandomForestRegressor()
                grid = GridSearchCV(clf, param_grid=param_lst, cv=2)
                grid.fit(in_x_train, in_y_train)
                results = grid.cv_results_
                best_fit = np.argmax(results.get("mean_test_score"))
                r2 = results.get("mean_test_score")
                get_params = results.get("params")[best_fit]
                param_inner.append([get_params])
                score_inner.append([r2])
                print(r2, get_params)
            score_outer = list(map(lambda x: np.mean(x), score_inner))
            best_case = np.argmax(score_outer)
            param_outer = list(map(lambda x: x, param_inner[best_case]))
            print(param_outer)
            clf = RandomForestRegressor(n_estimators=param_outer[0]["n_estimators"])
            clf.fit(out_x_train, out_y_train)
            r2 = clf.score(out_x_test, out_y_test)
            r2_rf.append(r2)
        return np.average(r2_rf)