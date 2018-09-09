from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import random
import numpy

from Utilities.SafeCastUtil import SafeCastUtil
from CustomModels.RecursiveBooleanPhrase import RecursiveBooleanPhrase
from CustomModels.ModelPhraseDataObject import ModelPhraseDataObject


class RandomSubsetElasticNetModel:

    # The number of times the model attempts to generate a boolean statement which partitions the features into a bucket
    # smaller than the upper bound and larger than the lower bound. Prevents recursion.
    MAX_BOOLEAN_GENERATION_ATTEMPTS = 10

    # The percentage of training data that needs to have at least one boolean phrase which matches it in order for
    # training to be complete.
    DEFAULT_COVERAGE_THRESHOLD = 0.8

    def __init__(self, alpha, l_one_ratio, binary_feature_indices, upper_bound=0.35, lower_bound=0.10, p=0,
                 explicit_model_count=-1):
        self.alpha = alpha
        self.l_one_ratio = l_one_ratio
        self.feature_indices_to_values = self.determineBinCatFeatureIndices(binary_feature_indices)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.p = p
        self.explicit_model_count = explicit_model_count  # Explicit count of models to create
        self.models_by_phrase = []
        self.fallback_model = None

    def determineBinCatFeatureIndices(self, binary_feature_indices):
        bin_cat_features = {}
        for i in range(0, len(binary_feature_indices)):
            bin_cat_features[i] = []  # Filled in at fitting time.
        return bin_cat_features

    def fit(self, features, results):
        self.determineUniqueFeatureBinaryFeatureValues(features)
        min_count = SafeCastUtil.safeCast(self.lower_bound * len(features), int)
        if min_count == 0:
            min_count = 1
        max_count = SafeCastUtil.safeCast(self.upper_bound * len(features), int)

        matching_threshold = SafeCastUtil.safeCast(self.DEFAULT_COVERAGE_THRESHOLD * len(features), int)
        full_pool = {"features": features[:], "results": results[:]}

        total_matched_feature_sets = 0
        rounds_with_no_new_matches = 0
        boolean_generation_attempts = 0

        while (((len(self.models_by_phrase) < self.explicit_model_count) and self.explicit_model_count > 0) or
               (total_matched_feature_sets < matching_threshold and self.explicit_model_count <= 0)) and\
                rounds_with_no_new_matches < self.MAX_BOOLEAN_GENERATION_ATTEMPTS:
            unused_features = SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list)
            current_phrase = None
            match_pool = {"features": [], "results": []}

            while (len(match_pool["features"]) < min_count or len(match_pool["features"]) > max_count) \
                    and boolean_generation_attempts < self.MAX_BOOLEAN_GENERATION_ATTEMPTS:
                if len(unused_features) == 0:
                    boolean_generation_attempts += 1
                    break

                feature_to_split_on = random.choice(unused_features)
                unused_features = [feature for feature in unused_features if feature is not feature_to_split_on]
                current_phrase = self.generatePhrase(current_phrase, feature_to_split_on, min_count, match_pool)

                match_pool = self.fetchMatchingPool(current_phrase, full_pool)

                if min_count <= len(match_pool["features"]) <= max_count:
                    self.createAndFitModel(current_phrase, match_pool)
                    new_matched_feature_sets = len(self.featuresMatchingPhrase(features))
                    if new_matched_feature_sets <= total_matched_feature_sets:
                        rounds_with_no_new_matches += 1
                    else:
                        rounds_with_no_new_matches = 0
                    total_matched_feature_sets = new_matched_feature_sets

            if boolean_generation_attempts >= self.MAX_BOOLEAN_GENERATION_ATTEMPTS:
                break

        fallback_phrase = RecursiveBooleanPhrase(None, None, None, None)
        self.createAndFitModel(fallback_phrase, full_pool)

    def determineUniqueFeatureBinaryFeatureValues(self, features):
        for feature_set in features:
            for i in range(0, len(feature_set)):
                if i in SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list) and feature_set[i] not in \
                        self.feature_indices_to_values[i]:
                    self.feature_indices_to_values[i].append(feature_set[i])

    def generatePhrase(self, current_phrase, feature_to_split_on, min_count, selected_pool):
        value_to_split_on = random.choice(self.feature_indices_to_values[feature_to_split_on])
        is_or = len(selected_pool["features"]) < min_count
        return RecursiveBooleanPhrase(feature_to_split_on, value_to_split_on, is_or, current_phrase)

    def fetchMatchingPool(self, current_phrase, full_pool):
        match_pool = {"features": [], "results": []}
        for i in range(0, len(full_pool["features"])):
            feature_set = full_pool["features"][i]
            result = full_pool["results"][i]
            if current_phrase.analyzeForFeatureSet(feature_set):
                match_pool["features"].append(feature_set)
                match_pool["results"].append(result)
        return match_pool

    def createAndFitModel(self, current_phrase, selected_pool):
        trimmed_features = self.trimBooleanFeatures(selected_pool["features"])
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l_one_ratio)
        model.fit(trimmed_features, selected_pool["results"])
        r_squared_score = r2_score(selected_pool["results"], model.predict(trimmed_features))
        if current_phrase.split is None or r_squared_score <= 0:  # always accept the fallback model
            model_phrase = ModelPhraseDataObject(model, current_phrase, r_squared_score)
            self.models_by_phrase.append(model_phrase)

    def trimBooleanFeatures(self, features):
        trimmed_features = []
        for feature_set in features:
            trimmed_feature_set = []
            for i in range(0, len(feature_set)):
                if i not in SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list):
                    trimmed_feature_set.append(feature_set[i])
            trimmed_features.append(trimmed_feature_set)
        return trimmed_features

    def featuresMatchingPhrase(self, features):
        matching_features = []
        for feature_set in features:
            matches_a_phrase = False
            for model_by_phrase in self.models_by_phrase:
                if model_by_phrase.phrase.analyzeForFeatureSet(feature_set):
                    matches_a_phrase = True
                    break
            if matches_a_phrase:
                matching_features.append(feature_set)

        return matching_features

    def predict(self, features):
        predictions = []  # Order matters, so we can't sort into pools.
        for feature in features:
            predictions_and_scores = []
            for model_by_phrase in self.models_by_phrase:
                if model_by_phrase.phrase.analyzeForFeatureSet(feature):
                    raw_prediction = model_by_phrase.model.predict(self.trimBooleanFeatures([feature]))
                    predictions_and_scores.append([raw_prediction[0], model_by_phrase.score])

            sum_of_matching_r2 = numpy.sum([pred[1] for pred in predictions_and_scores])
            if sum_of_matching_r2 > 0:
                max_weight = numpy.max([pred[1] for pred in predictions_and_scores])
                num_weights_with_max_weight = len([pred[1] for pred in predictions_and_scores if pred[1] == max_weight])

                matching_weighted_preds = []
                for i in range(0, len(predictions_and_scores)):
                    is_max = 0
                    if predictions_and_scores[i][1] == max_weight:
                        is_max = 1
                    ratio_of_max_score = predictions_and_scores[i][1] / sum_of_matching_r2

                    weight = (self.p * is_max / num_weights_with_max_weight) + ((1 - self.p) * ratio_of_max_score)
                    matching_weighted_preds.append(weight * predictions_and_scores[i][0])

                predictions.append(numpy.sum(matching_weighted_preds))
            else:
                unweighted_pred = numpy.average([pred[0] for pred in predictions_and_scores])
                predictions.append(unweighted_pred)

        return predictions

    def score(self, features, relevant_results):
        return r2_score(relevant_results, self.predict(features))
