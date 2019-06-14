from sklearn.linear_model import Ridge
from collections import OrderedDict

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RidgeRegressionTrainer(AbstractModelTrainer):

    ALPHA = "alpha"

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RIDGE_REGRESSION, self.initializeHyperParameters(),
                         is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        hyperparams = OrderedDict()
        hyperparams[self.ALPHA] = [0.01, 0.1, 1, 10]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        model = Ridge(alpha=hyperparams.get(self.ALPHA), normalize=True)
        model.fit(features, results)
        self.log.debug("Successful creation of the Ridge Regression model: %s\n", model)
        return model

    def fetchFeatureImportances(self, model, features_in_order):
        if hasattr(model, "coef_") and hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
            return super().normalizeCoefficients(model.coef_, features_in_order)

        return {}
