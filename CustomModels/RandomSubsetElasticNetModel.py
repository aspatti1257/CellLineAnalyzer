from sklearn.linear_model import ElasticNet
import random

from Utilities.SafeCastUtil import SafeCastUtil
from CustomModels.RecursiveBooleanPhrase import RecursiveBooleanPhrase
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class RandomSubsetElasticNetModel:

    # The number of times the model attempts to generate a boolean statement which partitions the features into a bucket
    # smaller than the upper bound and larger than the lower bound. Prevents recursion.
    MAX_BOOLEAN_GENERATION_ATTEMPTS = 10

    def __init__(self, upper_bound, lower_bound, alpha, l_one_ratio, feature_names, bin_cat_matrix_name):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.alpha = alpha
        self.l_one_ratio = l_one_ratio
        self.feature_names = feature_names
        self.bin_cat_matrix_name = bin_cat_matrix_name
        self.feature_indices_to_values = self.determineBinCatFeatureIndices(feature_names, bin_cat_matrix_name)
        self.models_by_statement = []
        self.fallback_model = None

    def determineBinCatFeatureIndices(self, feature_names, bin_cat_matrix_name):
        bin_cat_features = {}
        for i in range(0, len(feature_names)):
            if bin_cat_matrix_name in feature_names[i]:
                bin_cat_features[i] = []  # Filled in at fitting time.
        return bin_cat_features

    def fit(self, features, results):
        self.determineUniqueFeatureBinaryFeatureValues(features)
        min_count = SafeCastUtil.safeCast(self.lower_bound * len(features), int)
        max_count = SafeCastUtil.safeCast(self.upper_bound * len(features), int)

        current_pool = {"features": features[:], "results": results[:]}

        boolean_generation_attempts = 0

        while len(current_pool["features"]) > min_count:

            match_pool = {"features": [], "results": []}
            current_phrase = None
            unused_features = SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list)

            while (len(match_pool["features"]) < min_count or len(match_pool["features"]) > max_count) \
                    and boolean_generation_attempts < self.MAX_BOOLEAN_GENERATION_ATTEMPTS:
                if len(unused_features) == 0:
                    boolean_generation_attempts += 1
                    break

                feature_to_split_on = random.choice(unused_features)
                unused_features = [feature for feature in unused_features if feature is not feature_to_split_on]
                current_phrase = self.generatePhrase(current_phrase, feature_to_split_on, min_count, match_pool)

                remaining_pool, match_pool = self.sortIntoPoolsByPhrase(current_phrase, current_pool)

                if min_count < len(match_pool["features"]) < max_count:
                    self.createAndFitModel(current_phrase, match_pool)
                    current_pool = remaining_pool
            if boolean_generation_attempts >= self.MAX_BOOLEAN_GENERATION_ATTEMPTS:
                break

        self.fallback_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l_one_ratio)
        self.fallback_model.fit(self.trimBooleanFeatures(features), results)

    def determineUniqueFeatureBinaryFeatureValues(self, features):
        for feature_set in features:
            for i in range(0, len(feature_set)):
                if i in SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list) and feature_set[i] not in \
                        self.feature_indices_to_values[i]:
                    self.feature_indices_to_values[i].append(feature_set[i])

    def generatePhrase(self, current_phrase, feature_to_split_on, min_count, selected_pool):
        value_to_split_on = random.choice(self.feature_indices_to_values[feature_to_split_on])
        feature_name = self.feature_names[feature_to_split_on]
        is_or = len(selected_pool["features"]) < min_count
        return RecursiveBooleanPhrase(feature_to_split_on, feature_name, value_to_split_on, is_or, current_phrase)

    def sortIntoPoolsByPhrase(self, current_phrase, current_pool):
        match_pool = {"features": [], "results": []}
        remaining_pool = {"features": [], "results": []}
        for i in range(0, len(current_pool["features"])):
            feature_set = current_pool["features"][i]
            result = current_pool["results"][i]
            if current_phrase.analyzeForFeatureSet(feature_set):
                match_pool["features"].append(feature_set)
                match_pool["results"].append(result)
            else:
                remaining_pool["features"].append(feature_set)
                remaining_pool["results"].append(result)
        return remaining_pool, match_pool

    def createAndFitModel(self, current_phrase, selected_pool):
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l_one_ratio)
        trimmed_features = self.trimBooleanFeatures(selected_pool["features"])
        model.fit(trimmed_features, selected_pool["results"])
        self.models_by_statement.append({
            "phrase": current_phrase,
            "model": model
        })

    def trimBooleanFeatures(self, features):
        trimmed_features = []
        for feature_set in features:
            trimmed_feature_set = []
            for i in range(0, len(feature_set)):
                if i not in SafeCastUtil.safeCast(self.feature_indices_to_values.keys(), list):
                    trimmed_feature_set.append(feature_set[i])
            trimmed_features.append(trimmed_feature_set)
        return trimmed_features

    def predict(self, features):
        predictions = []  # Order matters, so we can't sort into pools.
        for feature in features:
            prediction = None
            for boolean_statement in self.models_by_statement:
                if boolean_statement["phrase"].analyzeForFeatureSet(feature):
                    prediction = boolean_statement["model"].predict(self.trimBooleanFeatures([feature]))
                    break
            if prediction is None:
                prediction = self.fallback_model.predict(self.trimBooleanFeatures([feature]))
            predictions.append(prediction[0])
        return predictions

    def score(self, features, relevant_results):
        max_score = AbstractModelTrainer.DEFAULT_MIN_SCORE

        current_pool = {"features": features[:], "results": relevant_results[:]}
        for model_by_statement in self.models_by_statement:
            remaining_pool, match_pool = self.sortIntoPoolsByPhrase(model_by_statement.get("phrase"), current_pool)

            max_score = self.fetchBestScoreByModel(match_pool.get("features"), match_pool.get("results"),
                                                   max_score, model_by_statement.get("model"))
            current_pool = remaining_pool

        return self.fetchBestScoreByModel(current_pool.get("features"), current_pool.get("results"),
                                          max_score, self.fallback_model)

    def fetchBestScoreByModel(self, features, results, current_score, model):
        trimmed_features = self.trimBooleanFeatures(features)
        if len(trimmed_features) > 0:
            score = model.score(trimmed_features, results)
            if score > current_score:
                return score
        return current_score

