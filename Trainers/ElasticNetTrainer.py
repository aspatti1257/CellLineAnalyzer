from sklearn.linear_model import ElasticNet

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class ElasticNetTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.ELASTIC_NET, self.initializeHyperParameters(), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        return {
            "alpha": [0.01, 0.1, 1, 10],
            "l_one_ratio": [0, 0.1, 0.5, 0.9, 1]
        }

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        if self.is_classifier:
            self.log.debug("Unable to train Elastic Net classifier. Returning default min score.")
            return None
        else:
            model = ElasticNet(alpha=hyperparams[0], l1_ratio=hyperparams[1])
        model.fit(features, results)
        self.log.debug("Successful creation of Elastic Net model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], hyperparam_set[1]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                      "alpha = %s\n" +
                      "l one ratio = %s", feature_set_as_string, self.algorithm, hyperparams[0],
                      hyperparams[1])

    def fetchFeatureImportances(self, model, gene_list_combo):
        features_in_order = super().generateFeaturesInOrder(gene_list_combo)
        if hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
            return super().normalizeCoefficients(model.coef_, features_in_order)

        return {}
