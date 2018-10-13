from sklearn.linear_model import ElasticNet
from collections import OrderedDict

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class ElasticNetTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.ELASTIC_NET, self.initializeHyperParameters(), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        hyperparams = OrderedDict()
        hyperparams["alpha"] = [0.01, 0.1, 1, 10]
        hyperparams["l_one_ratio"] = [0, 0.1, 0.5, 0.9, 1]
        return hyperparams

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

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string, record_diagnostics, input_folder):
        message = "Optimal Hyperparameters for " + feature_set_as_string + " " + self.algorithm + " algorithm " \
                  "chosen as:\n" +\
                        "\talpha = " + SafeCastUtil.safeCast(hyperparams[0], str) + "\n" \
                        "\tl_one_ratio = " + SafeCastUtil.safeCast(hyperparams[1], str) + ".\n"
        self.log.info(message)
        if record_diagnostics:
            self.writeToDiagnosticsFile(input_folder, message)

    def fetchFeatureImportances(self, model, gene_list_combo):
        features_in_order = super().generateFeaturesInOrder(gene_list_combo)
        if hasattr(model, "coef_") and len(features_in_order) == len(model.coef_):
            return super().normalizeCoefficients(model.coef_, features_in_order)

        return {}
