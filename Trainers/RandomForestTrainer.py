import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class RandomForestTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_FOREST, self.initializeHyperParameters(0, 0), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self, n, p):
        return {
            "max_depth": [0.05 * n, 0.1 * n, 0.2 * n, 0.3 * n, 0.4 * n, 0.5 * n, 0.75 * n, 1 * n],
            "m_val": [1, (1 + numpy.sqrt(p)) / 2, numpy.sqrt(p), (numpy.sqrt(p) + p) / 2, p]
        }

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        n = len(SafeCastUtil.safeCast(training_matrix.keys(), list))  # number of samples
        p = len(SafeCastUtil.safeCast(training_matrix.values(), list)[0])  # number of features
        return super().loopThroughHyperparams(self.initializeHyperParameters(n, p), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        max_depth = numpy.min([SafeCastUtil.safeCast(numpy.floor(hyperparams[0]), int), len(features)])
        max_leaf_nodes = numpy.maximum(2, SafeCastUtil.safeCast(numpy.ceil(hyperparams[1]), int))
        if self.is_classifier:
            model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)
        else:
            model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)
        model.fit(features, results)
        self.log.debug("Successful creation of Random Forest model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], hyperparam_set[1]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                      "m_val = %s\n" +
                      "max depth = %s", feature_set_as_string, self.algorithm, hyperparams[0], hyperparams[1])

    def fetchFeatureImportances(self, model, gene_list_combo):
        importances = {}
        features_in_order = super().generateFeaturesInOrder(gene_list_combo)

        if hasattr(model, "feature_importances_") and len(features_in_order) == len(model.feature_importances_):
            for i in range(0, len(features_in_order)):
                importances[features_in_order[i]] = model.feature_importances_[i]  # already normalized.

        return importances
