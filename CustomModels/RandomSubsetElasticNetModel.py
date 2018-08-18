from sklearn.linear_model import ElasticNet


class RandomSubsetElasticNetModel:

    feature_importances_ = None

    boolean_statements = []

    def __init__(self, upper_bound, lower_bound, alpha, l_one_ratio, binary_categorical_features):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.alpha = alpha
        self.l_one_ratio = l_one_ratio
        self.binary_categorical_features = binary_categorical_features

    def fit(self, features, results):
        # TODO: For each partition, there needs to be a boolean statement which can be used to partition the test data
        # properly. However, in order to support test data that doesn't fit into any boolean statement, we should
        # support a "catch all" model. After all of the remaining partitions are below a certain value (probably the
        # minimum partition size), train a model with the ENTIRE dataset. Then, any test data that doesn't fit this
        # any existing boolean statement can still be tested against this model.

        pass

    def predict(self, features):
        pass  # TODO

    def score(self, features, relevant_results):
        pass  # TODO

