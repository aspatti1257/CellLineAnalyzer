from sklearn.linear_model import ElasticNet
from collections import OrderedDict

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class ElasticNetTrainer(AbstractModelTrainer):

    ALPHA = "alpha"
    L_ONE_RATIO = "l_one_ratio"

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.ELASTIC_NET, self.initializeHyperParameters(), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        hyperparams = OrderedDict()
        hyperparams[self.ALPHA] = [0.01, 0.1, 1, 10]
        hyperparams[self.L_ONE_RATIO] = [0, 0.1, 0.5, 0.9, 1]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        if self.is_classifier:
            self.log.debug("Unable to train Elastic Net classifier. Returning default min score.")
            return None
        else:
            model = ElasticNet(alpha=hyperparams.get(self.ALPHA), l1_ratio=hyperparams.get(self.L_ONE_RATIO))
        model.fit(features, results)
        self.log.debug("Successful creation of Elastic Net model: %s\n", model)
        return model

    def fetchFeatureImportances(self, model, features_in_order):
        if hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
            return super().normalizeCoefficients(model.coef_, features_in_order)

        return {}
