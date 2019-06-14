import numpy
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class RandomForestTrainer(AbstractModelTrainer):

    MAX_FEATURES = "max_features"
    MIN_SAMPLES_SPLIT = "min_samples_split"

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_FOREST, self.initializeHyperParameters(0), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self, p):
        hyperparams = OrderedDict()
        hyperparams[self.MAX_FEATURES] = [numpy.sqrt(p), (numpy.sqrt(p) + p) / 2, p]
        hyperparams[self.MIN_SAMPLES_SPLIT] = [2, 10, 20]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        p = len(SafeCastUtil.safeCast(training_matrix.values(), list)[0])  # number of features
        return super().loopThroughHyperparams(self.initializeHyperParameters(p), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        max_features = numpy.min([SafeCastUtil.safeCast(numpy.floor(hyperparams.get(self.MAX_FEATURES)), int), len(features[0])])

        min_samples_split = hyperparams.get(self.MIN_SAMPLES_SPLIT)
        if min_samples_split > 1:
            min_samples_split = SafeCastUtil.safeCast(min_samples_split, int)
        if self.is_classifier:
            model = RandomForestClassifier(n_estimators=100, min_samples_split=min_samples_split, max_features=max_features)
        else:
            model = RandomForestRegressor(n_estimators=100, min_samples_split=min_samples_split, max_features=max_features)
        model.fit(features, results)
        self.log.debug("Successful creation of Random Forest model: %s\n", model)
        return model

    def fetchFeatureImportances(self, model, features_in_order):
        importances = {}

        if hasattr(model, "feature_importances_") and len(features_in_order) == len(model.feature_importances_):
            for i in range(0, len(features_in_order)):
                importances[features_in_order[i]] = model.feature_importances_[i]  # already normalized.

        return importances
