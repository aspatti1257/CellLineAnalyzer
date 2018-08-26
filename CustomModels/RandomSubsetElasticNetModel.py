from sklearn.linear_model import ElasticNet
import random

from CustomModels.RecursiveBooleanPhrase import RecursiveBooleanPhrase


class RandomSubsetElasticNetModel:

    def __init__(self, upper_bound, lower_bound, alpha, l_one_ratio, feature_names, bin_cat_matrix_name):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.alpha = alpha
        self.l_one_ratio = l_one_ratio
        self.feature_names = feature_names
        self.bin_cat_matrix_name = bin_cat_matrix_name
        self.bin_cat_feature_indices = self.determineBinCatFeatureIndices(feature_names, bin_cat_matrix_name)
        self.models_by_statement = []
        self.fallback_model = None

    def determineBinCatFeatureIndices(self, feature_names, bin_cat_matrix_name):
        bin_cat_feature_indices = []
        for i in range(0, len(feature_names)):
            if bin_cat_matrix_name in feature_names[i]:
                bin_cat_feature_indices.append(i)
        return bin_cat_feature_indices

    def fit(self, features, results):
        current_pool = {"features": features[:], "results": results[:]}

        while len(current_pool["features"]) > self.lower_bound:

            selected_pool = {"features": [], "results": []}
            current_phrase = None
            unused_features = self.bin_cat_feature_indices

            while len(selected_pool["features"]) < self.lower_bound or len(selected_pool["features"]) > self.upper_bound:
                if len(unused_features) == 0:
                    break

                feature_to_split_on = random.choice(unused_features)
                unused_features = [feature for feature in unused_features if feature is not feature_to_split_on]
                value_to_split_on = random.choice([0, 1])

                current_phrase = RecursiveBooleanPhrase(feature_to_split_on, value_to_split_on,
                                                        len(selected_pool["features"]) < self.lower_bound, current_phrase)

                selected_pool = {"features": [], "results": []}
                remaining_pool = {"features": [], "results" : []}
                for i in range(0, len(current_pool["features"])):
                    feature_set = current_pool["features"][i]
                    result = current_pool["results"][i]
                    if current_phrase.analyzeForFeatureSet(feature_set):
                        selected_pool["features"].append(feature_set)
                        selected_pool["results"].append(result)
                    else:
                        remaining_pool["features"].append(feature_set)
                        remaining_pool["results"].append(result)

                if self.lower_bound < len(selected_pool["features"]) < self.upper_bound:
                    model = ElasticNet(alpha=self.alpha, l1_ratio=self.l_one_ratio)
                    trimmed_features = self.trimBooleanFeatures(selected_pool["features"])
                    model.fit(trimmed_features, selected_pool["results"])
                    self.models_by_statement.append({
                        "phrase": current_phrase,
                        "model": model,
                        "features": trimmed_features,
                        "results": selected_pool["results"]
                    })
                    current_pool = remaining_pool

        self.fallback_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l_one_ratio)
        self.fallback_model.fit(self.trimBooleanFeatures(features), results)

        # TODO: For each partition, there needs to be a boolean statement which can be used to partition the test data
        # properly. However, in order to support test data that doesn't fit into any boolean statement, we should
        # support a "catch all" model. After all of the remaining partitions are below a certain value (probably the
        # minimum partition size), train a model with the ENTIRE dataset. Then, any test data that doesn't fit this
        # any existing boolean statement can still be tested against this model.

    def trimBooleanFeatures(self, features):
        trimmed_features = []
        for feature_set in features:
            trimmed_feature_set = []
            for i in range(0, len(feature_set)):
                if i not in self.bin_cat_feature_indices:
                    trimmed_feature_set.append(feature_set[i])
            trimmed_features.append(trimmed_feature_set)
        return trimmed_features

    def predict(self, features):
        predictions = []
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
        scores = []
        for i in range(0, len(features)):
            score = None
            for boolean_statement in self.models_by_statement:
                if boolean_statement["phrase"].analyzeForFeatureSet(features[i]):
                    score = boolean_statement["model"].score(self.trimBooleanFeatures([features[i]]), [relevant_results[i]])
                    break
            if score is None:
                score = self.fallback_model.score(self.trimBooleanFeatures([features[i]]), [relevant_results[i]])
            scores.append(score)  # TODO: Investigate why this is always 0
        return scores

