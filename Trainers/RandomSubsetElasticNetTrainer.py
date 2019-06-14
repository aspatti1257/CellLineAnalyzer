import numpy
import copy
from collections import OrderedDict

from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil
from CustomModels.RandomSubsetElasticNet import RandomSubsetElasticNet


class RandomSubsetElasticNetTrainer(AbstractModelTrainer):

    ALPHA = "alpha"
    L_ONE_RATIO = "l_one_ratio"

    def __init__(self, is_classifier, binary_categorical_matrix, p_val, k_val):
        self.validateBinaryCategoricalMatrix(binary_categorical_matrix)

        self.binary_categorical_matrix = binary_categorical_matrix
        # TODO: Can potentially break here if "." is in feature file name.
        self.bin_cat_matrix_name = binary_categorical_matrix.get(ArgumentProcessingService.FEATURE_NAMES)[0].split(".")[0]
        self.p_val = p_val
        self.k_val = k_val
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
            self.log.debug("Valid binary categorical matrix.")
        else:
            raise ValueError("Invalid binary categorical matrix.")

    def supportsHyperparams(self):
        return True

    def preserveNonHyperparamData(self, model_data, model):
        if model_data.get(self.ADDITIONAL_DATA) is None:
            model_data[self.ADDITIONAL_DATA] = []
        for model_phrase in model.models_by_phrase:
            model_data[self.ADDITIONAL_DATA].append(model_phrase)

    def initializeHyperParameters(self):
        hyperparams = OrderedDict()
        hyperparams[self.ALPHA] = [0.01, 0.1, 1, 10]
        hyperparams[self.L_ONE_RATIO] = [0, 0.1, 0.5, 0.9, 1]
        return hyperparams

    def hyperparameterize(self, training_matrix, testing_matrix, results):
        return super().loopThroughHyperparams(self.initializeHyperParameters(), training_matrix, testing_matrix, results)

    def train(self, results, features, hyperparams, feature_names):
        binary_feature_indices = self.fetchBinaryFeatureIndices(feature_names)
        model = RandomSubsetElasticNet(hyperparams.get(self.ALPHA), hyperparams.get(self.L_ONE_RATIO),
                                       binary_feature_indices, p=self.p_val,
                                       explicit_phrases=self.determineExplicitPhrases(hyperparams))

        model.fit(features, results)
        self.log.debug("Successful creation of Random Subset Elastic Net model: %s\n", model)
        return model

    def fetchBinaryFeatureIndices(self, feature_names):
        binary_feature_indices = []
        for i in range(0, len(feature_names)):
            if self.bin_cat_matrix_name in feature_names[i]:
                binary_feature_indices.append(i)
        return binary_feature_indices

    def determineExplicitPhrases(self, hyperparams):
        if len(hyperparams) < 3:
            return None
        phrase_sets = hyperparams.get(AbstractModelTrainer.ADDITIONAL_DATA)
        all_phrases_and_r2_scores = []
        for phrase_set in phrase_sets:
            for model_phrase in phrase_set:
                phrase_exists = False
                for existing_phrase in all_phrases_and_r2_scores:
                    if existing_phrase.phrase.equals(model_phrase.phrase):
                        phrase_exists = True
                        if model_phrase.score > existing_phrase.score:
                            existing_phrase.score = model_phrase.score
                if not phrase_exists:
                    all_phrases_and_r2_scores.append(model_phrase)
        ordered_phrases = sorted(all_phrases_and_r2_scores, key=lambda phrase: phrase.score, reverse=True)
        cutoff = numpy.max([SafeCastUtil.safeCast(len(ordered_phrases) * self.k_val, int), 1])

        return [model_phrase.phrase for model_phrase in ordered_phrases[:cutoff]]

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

    def fetchFeatureImportances(self, model, features_in_order):
        evaluated_features = [feature for feature in features_in_order if self.bin_cat_matrix_name not in feature]
        importances_map = OrderedDict()
        for model_phrase in model.models_by_phrase:
            if hasattr(model_phrase.model, "coef_") and len(evaluated_features) == len(model_phrase.model.coef_):
                for i in range(0, len(evaluated_features)):
                    weighted_score = model_phrase.model.coef_[i] * model_phrase.score
                    if importances_map.get(evaluated_features[i]) is None:
                        importances_map[evaluated_features[i]] = [weighted_score]
                    else:
                        importances_map[evaluated_features[i]].append(weighted_score)

        feature_names = SafeCastUtil.safeCast(importances_map.keys(), list)
        average_coefficients = [numpy.sum(imps) / len(evaluated_features) for imps in
                                SafeCastUtil.safeCast(importances_map.values(), list)]
        return super().normalizeCoefficients(average_coefficients, feature_names)

    def fetchModelPhrases(self, model, gene_list_combo):
        features_in_order = super().generateFeaturesInOrder(gene_list_combo)
        bin_cat_feature_indices = self.fetchBinaryFeatureIndices(features_in_order)
        index_to_feature_map = {}
        for index in bin_cat_feature_indices:
            index_to_feature_map[index] = features_in_order[index]
        return self.summarizeModelPhrases(index_to_feature_map, model)

    def summarizeModelPhrases(self, index_to_feature_map, model):
        scores_by_string = {}
        for model_phrase in model.models_by_phrase:
            formatted_phrase = self.recursivelyFormatPhrase(copy.deepcopy(model_phrase.phrase), index_to_feature_map)
            scores_by_string[formatted_phrase.toSummaryString()] = model_phrase.score
        return scores_by_string

    def recursivelyFormatPhrase(self, phrase, index_to_feature_map):
        if phrase.split is None:
            return phrase
        phrase.split = index_to_feature_map[phrase.split]
        if phrase.nested_phrase is not None:
            self.recursivelyFormatPhrase(phrase.nested_phrase, index_to_feature_map)
        return phrase
