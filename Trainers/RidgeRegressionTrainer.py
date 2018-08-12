from sklearn.linear_model import Ridge
import numpy

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RidgeRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RIDGE_REGRESSION, self.initializeHyperParameters(),
                         is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        return {"alpha": [0.01, 0.1, 1, 10]}

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        model = Ridge(alpha=hyperparams[0], normalize=True)
        model.fit(features, results)
        self.log.debug("Successful creation of the Ridge Regression model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                      "alpha = %s\n", feature_set_as_string, self.algorithm, hyperparams[0])

    def fetchFeatureImportances(self, model, gene_list_combo):
        features_in_order = super().generateFeaturesInOrder(gene_list_combo)
        if hasattr(model, "coef_") and hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
            return super().normalizeCoefficients(model.coef_, features_in_order)

        return {}
