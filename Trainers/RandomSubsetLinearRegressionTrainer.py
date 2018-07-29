import numpy

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from ArgumentProcessingService import ArgumentProcessingService

class RandomSubsetLinearRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier, binary_categorical_matrix):
        # TODO: assert this is indeed binary here. If not, throw exception.
        self.binary_categorical_matrix = binary_categorical_matrix
        self.current_feature_set = []
        self.formatted_binary_matrix = []
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_SUBSET_LINEAR_REGRESSION,
                         self.initializeHyperParameters(), is_classifier)

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        return {
            "upper_bound": [500, 300, 100],
            "lower_bound": [10, 30, 50]
        }

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        # TODO: create a new matrix consisting only of the features shared by self.binary_categorical_matrix.
        # Set them as a trainer level variable (self.formatted_binary_matrix).

        # Remove this assertion and MachineLearningService.setVariablesOnTrainerInSpecialCases() when this is reliable.
        assert training_matrix.get(ArgumentProcessingService.FEATURE_NAMES) == numpy.concatenate(self.current_feature_set).tolist()

        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams):
        # TODO: Split data and return a model. Must be able to implement "predict" and "score" methods correctly.
        return None

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], hyperparam_set[1]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # No hyperparams for this model
