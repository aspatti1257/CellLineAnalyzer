from sklearn.linear_model import Lasso
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class LassoRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.LASSO_REGRESSION, self.initializeHyperParameters(),
                         is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        return {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        model = Lasso(alpha=hyperparams[0], normalize=True, max_iter=0)
        model.fit(features, results)
        self.log.debug("Successful creation of the Lasso Regression model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                      "alpha = %s\n", feature_set_as_string, self.algorithm, hyperparams[0])