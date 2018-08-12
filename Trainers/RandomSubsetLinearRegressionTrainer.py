import numpy

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil


class RandomSubsetLinearRegressionTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier, binary_categorical_matrix):
        # TODO: assert this is indeed binary here. If not, throw exception.
        self.validateBinaryCategoricalMatrix(binary_categorical_matrix)

        self.binary_categorical_matrix = binary_categorical_matrix
        self.current_feature_set = []
        self.formatted_binary_matrix = []
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_SUBSET_LINEAR_REGRESSION,
                         self.initializeHyperParameters(), is_classifier)

    def validateBinaryCategoricalMatrix(self, binary_categorical_matrix):
        counter_dictionary = {}
        for key in binary_categorical_matrix.keys():
            if key == ArgumentProcessingService.FEATURE_NAMES:
                continue
            for value in binary_categorical_matrix[key]:
                if counter_dictionary.get(value) is None:
                    counter_dictionary[value] = 1
                else:
                    counter_dictionary[value] += 1
        is_valid = len(SafeCastUtil.safeCast(counter_dictionary.keys(), list)) == 2
        if is_valid:
            self.log.info("Valid binary categorical matrix.")
        else:
            raise ValueError("Invalid binary categorical matrix.")

    def supportsHyperparams(self):
        return True

    def initializeHyperParameters(self):
        return {
            "upper_bound": [500, 300, 100],
            "lower_bound": [10, 30, 50],
            "alpha": [0.1, 1, 10]
        }

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        # TODO: create a new matrix consisting only of the features shared by self.binary_categorical_matrix.
        # Set them as a trainer level variable (self.formatted_binary_matrix).

        # Remove this assertion and MachineLearningService.setVariablesOnTrainerInSpecialCases() when this is reliable.
        assert training_matrix.get(ArgumentProcessingService.FEATURE_NAMES) == numpy.concatenate(self.current_feature_set).tolist()

        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams):
        # TODO: Split data and return a model. Must be able to implement "predict" and "score" methods correctly.

        # TODO: For each partition, there needs to be a boolean statement which can be used to partition the test data
        # properly. However, in order to support test data that doesn't fit into any boolean statement, we should
        # support a "catch all" model. After all of the remaining partitions are below a certain value (probably the
        # minimum partition size), train a model with the random remainders. Then, any test data that doesn't fit this
        # any existing boolean statement can still be tested against this model.
        return None

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], hyperparam_set[1], hyperparam_set[2]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # No hyperparams for this model


    def fetchFeatureImportances(self, model, gene_list_combo):
        # TODO: Fetch feature importances from existing model object.
        return {}
