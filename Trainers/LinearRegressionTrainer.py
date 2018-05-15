from sklearn.linear_model import LinearRegression
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class LinearRegressionTrainer(AbstractModelTrainer):
    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.LINEAR_REGRESSION, self.initializeHyperParameters(),
                         is_classifier)

    def supportsHyperparams(self):
        return False

    def initializeHyperParameters(self):
        return {"bogus": [False]}

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        model = LinearRegression(copy_X=True)
        model.fit(features, results)
        self.log.debug("Successful creation of the Linear Regression model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[None, None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # No hyperparams for this model
