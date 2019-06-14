from sklearn import svm
from collections import OrderedDict

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class LinearSVMTrainer(AbstractModelTrainer):

    C_VAL = "c_val"
    EPSILON = "epsilon"

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.LINEAR_SVM, self.initializeHyperParameters(is_classifier),
                         is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self, is_classifier):
        hyperparams = OrderedDict()
        hyperparams[self.C_VAL] = [10E-2, 10E-1, 10E0]
        if not is_classifier:
            hyperparams[self.EPSILON] = [0.01, 0.05, 0.1, 0.15, 0.2]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(self.is_classifier), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        if self.is_classifier:
            model = svm.LinearSVC(C=hyperparams.get(self.C_VAL))
        else:
            model = svm.LinearSVR(C=hyperparams.get(self.C_VAL), epsilon=hyperparams.get(self.EPSILON))
        model.fit(features, results)
        self.log.debug("Successful creation of Linear Support Vector Machine model: %s\n", model)
        return model

    def fetchFeatureImportances(self, model, features_in_order):
        if hasattr(model, "coef_") and hasattr(model, "coef_"):
            if self.is_classifier and len(features_in_order) == len(model.coef_[0]):
                # TODO: This is probably wrong. Find proper way to get importances for classifier case.
                return super().normalizeCoefficients(model.coef_[0], features_in_order)
            elif not self.is_classifier and len(features_in_order) == len(model.coef_):
                return super().normalizeCoefficients(model.coef_, features_in_order)
        return {}
