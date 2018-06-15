from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
import numpy
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.SafeCastUtil import SafeCastUtil


class LinearRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier, record_diagnostics, input_folder):
        self.record_diagnostics = record_diagnostics
        self.input_folder = input_folder
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
        model = LinearRegression(copy_X=True, normalize=True)
        model.fit(features, results)
        self.log.debug("Successful creation of the Linear Regression model: %s\n", model)

        should_use_ridge = False
        for j in range(0, len(model.coef_)):
            if numpy.absolute(model.coef_[j]) > 10000:
                self.analyzeOddFeatureByIndex(j, features)
                should_use_ridge = True

        if should_use_ridge:
            self.log.debug("Abnormal Linear regression model created with very high magnitude coefficients. Falling"
                           "back to Ridge regression to salvage model results.")
            model = Ridge(copy_X=True, normalize=True)
            model.fit(features, results)

        return model

    def analyzeOddFeatureByIndex(self, index, features):
        feature_is_entirely_zero = True
        problem_feature = []
        other_features = {}
        for feature in features:
            for i in range(0, len(feature)):
                if i == index:
                    problem_feature.append(feature[i])
                elif other_features.get(i) is not None:
                    other_features[i].append(feature[i])
                else:
                    other_features[i] = [feature[i]]
            if feature[index] != 0:
                feature_is_entirely_zero = False

        num_correlated_features = 0
        for other_feature in other_features.values():
            correlation = pearsonr(other_feature, problem_feature)
            if numpy.absolute(correlation[0]) == 1.0:
                num_correlated_features += 1

        message = None
        if feature_is_entirely_zero:
            message = "Linear Model feature is entirely zero.\n"
        elif num_correlated_features > 0:
            message = "Linear Model feature correlated with " + \
                      SafeCastUtil.safeCast(num_correlated_features, str) + " other features.\n"

        if message is not None:
            self.log.debug(message)
            if self.record_diagnostics and self.input_folder is not None:
                self.writeToDiagnosticsFile(self.input_folder, message)

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[None, None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # No hyperparams for this model
