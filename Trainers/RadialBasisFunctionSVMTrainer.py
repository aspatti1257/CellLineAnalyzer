from collections import OrderedDict

from sklearn import svm

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RadialBasisFunctionSVMTrainer(AbstractModelTrainer):

    C_VAL = "c_val"
    GAMMA = "gamma"
    EPSILON = "epsilon"

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM,
                         self.initializeHyperParameters(is_classifier), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self, is_classifier):
        hyperparams = OrderedDict()
        hyperparams[self.C_VAL] = [10E-2, 10E-1, 10E0]
        hyperparams[self.GAMMA] = [10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 10E0, 10E1]
        if not is_classifier:
            hyperparams[self.EPSILON] = [0.01, 0.05, 0.1, 0.15, 0.2]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(self.is_classifier), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        if len(hyperparams) == 2 or self.is_classifier:
            model = svm.SVC(kernel='rbf', C=hyperparams.get(self.C_VAL), gamma=hyperparams.get(self.GAMMA))
        else:
            model = svm.SVR(kernel='rbf', C=hyperparams.get(self.C_VAL), gamma=hyperparams.get(self.GAMMA),
                            epsilon=hyperparams.get(self.EPSILON))
        model.fit(features, results)
        self.log.debug("Successful creation of RBF Support Vector Machine model: %s\n", model)
        return model

    def fetchFeatureImportances(self, model, features_in_order):
        return {}  # Not supported.
