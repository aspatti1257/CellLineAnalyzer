from sklearn import svm

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RadialBasisFunctionSVMTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM,
                         self.initializeHyperParameters(is_classifier), is_classifier)

    def initializeHyperParameters(self, is_classifier):
        hyperparams = {
            "c_val": [10E-2, 10E-1, 10E0],
            "gamma": [10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 10E0, 10E1]
        }
        if not is_classifier:
            hyperparams["epsilon"] = [0.01, 0.05, 0.1, 0.15, 0.2]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(self.is_classifier), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        if len(hyperparams) == 2 or self.is_classifier:
            model = svm.SVC(kernel='rbf', C=hyperparams[0], gamma=hyperparams[1])
        else:
            model = svm.SVR(kernel='rbf', C=hyperparams[0], gamma=hyperparams[1], epsilon=hyperparams[2])
        model.fit(features, results)
        self.log.debug("Successful creation of RBF Support Vector Machine model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        if not self.is_classifier:
            model_data[hyperparam_set[0], hyperparam_set[1], hyperparam_set[2]] = current_model_score
        else:
            model_data[hyperparam_set[0], hyperparam_set[1], None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        if self.is_classifier:
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "c_val = %s\n" +
                          "gamma = %s", feature_set_as_string, self.algorithm, hyperparams[0],
                          hyperparams[1])
        else:
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "c_val = %s\n" +
                          "gamma = %s\n" +
                          "episolon = %s", feature_set_as_string, self.algorithm, hyperparams[0],
                          hyperparams[1], hyperparams[2])