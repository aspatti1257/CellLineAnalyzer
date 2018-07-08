from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RandomPartitionLinearRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier):
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_PARTITION_LINEAR_REGRESSION,
                         self.initializeHyperParameters(), is_classifier)

    def supportsHyperparams(self):
        return False  # This may not necessarily be true. There may be some hyperparams.

    def initializeHyperParameters(self):
        return {"bogus": [False]}

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
