import numpy

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil
from CustomModels.RandomSubsetElasticNetModel import RandomSubsetElasticNetModel


class RandomSubsetElasticNetTrainer(AbstractModelTrainer):

    def __init__(self, is_classifier, binary_categorical_matrix):
        self.validateBinaryCategoricalMatrix(binary_categorical_matrix)

        self.binary_categorical_matrix = binary_categorical_matrix
        self.bin_cat_matrix_name = binary_categorical_matrix.get(ArgumentProcessingService.FEATURE_NAMES)[0].split(".")[0]
        self.current_feature_set_as_strings = []
        self.formatted_binary_matrix = []
        super().__init__(SupportedMachineLearningAlgorithms.RANDOM_SUBSET_ELASTIC_NET,
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
            "upper_bound": [0.95, 0.85, 0.75],
            "lower_bound": [0.10, 0.15, 0.2], # TODO: Remove these
            "alpha": [0.001, 0.01, 0.1, 1, 10],
            "l_one_ratio": [0, 0.1, 0.5, 0.9, 1]
        }

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        binary_feature_indices = []
        for i in range(0, len(feature_names)):
            if self.bin_cat_matrix_name in feature_names[i]:
                binary_feature_indices.append(i)

        model = RandomSubsetElasticNetModel(hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3],
                                            binary_feature_indices)

        model.fit(features, results)
        self.log.debug("Successful creation of Random Subset Elastic Net model: %s\n", model)
        return model

    def setModelDataDictionary(self, model_data, hyperparam_set, current_model_score):
        model_data[hyperparam_set[0], hyperparam_set[1], hyperparam_set[2], hyperparam_set[3]] = current_model_score

    def logOptimalHyperParams(self, hyperparams, feature_set_as_string):
        pass  # TODO

    def shouldProcessFeatureSet(self, feature_set):
        # Feature set should contain a gene list applied to the binary categorical matrix AND another gene list applied
        # to another feature file. Otherwise, skip training for this combo.
        uses_bin_cat_matrix = False
        uses_other_feature_file = False
        for features_by_file in feature_set:
            if len([feature for feature in features_by_file if self.bin_cat_matrix_name in feature]) > 0:
                uses_bin_cat_matrix = True
            if len([feature for feature in features_by_file if self.bin_cat_matrix_name not in feature]) > 0:
                uses_other_feature_file = True

        return uses_bin_cat_matrix and uses_other_feature_file

    def fetchFeatureImportances(self, model, gene_list_combo):
        # TODO: Fetch feature importances from existing model object.
        return {}
