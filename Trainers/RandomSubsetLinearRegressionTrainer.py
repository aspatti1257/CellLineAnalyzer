from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RandomSubsetLinearRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier, binary_categorical_matrix):
        #TODO: Do we actually need to pass this in here?
        self.binary_categorical_matrix = binary_categorical_matrix
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
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix,
                                              testing_matrix, results)

    def train(self, results, features, hyperparams):
        # TODO: Split data and return a model. Must be able to implement "predict" and "score" methods correctly.
        return None

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[None, None] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # No hyperparams for this model
