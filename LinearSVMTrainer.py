from AbstractModelTrainer import AbstractModelTrainer
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from sklearn import svm


class LinearSVMTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.LINEAR_SVM, self.initializeHyperParameters(is_classifier),
                         is_classifier)

    def initializeHyperParameters(self, is_classifier):
        hyperparams = {
            "c_val": [10E-2, 10E-1, 10E0]
        }
        if not is_classifier:
            hyperparams["epsilon"] = [0.01, 0.05, 0.1, 0.15, 0.2]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(self.is_classifier), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        if self.is_classifier:
            if type(hyperparams) is float:
                model = svm.LinearSVC(C=hyperparams)
            else:
                model = svm.LinearSVC(C=hyperparams[0])
        else:
            model = svm.LinearSVR(C=hyperparams[0], epsilon=hyperparams[1])
        model.fit(features, results)
        self.log.debug("Successful creation of Linear Support Vector Machine model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        if not self.is_classifier:
            model_data[hyperparam_set[0], hyperparam_set[1]] = current_model_score
        else:
            model_data[hyperparam_set[0]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        if self.is_classifier:
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "c_val = %s\n", feature_set_as_string, self.algorithm, hyperparams)
        else:
            self.log.info("Optimal Hyperparameters for %s %s algorithm chosen as:\n" +
                          "c_val = %s\n" +
                          "epsilon = %s", feature_set_as_string, self.algorithm, hyperparams[0], hyperparams[1])
