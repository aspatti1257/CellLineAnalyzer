import unittest
import logging
import random
import copy


from CustomModels.RandomSubsetElasticNet import RandomSubsetElasticNet
from Utilities.SafeCastUtil import SafeCastUtil
from CustomModels.RecursiveBooleanPhrase import RecursiveBooleanPhrase


class RandomSubsetElasticNetModelTest(unittest.TestCase):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    train_features = [[0, 0, 0, 1, 0.32, 0.25, 0.52, 0.63],
                      [0, 0, 1, 1, 1.11, 1.45, 0.31, 0.22],
                      [0, 1, 0, 0, 0.32, 0.56, 0.66, 0.25],
                      [1, 0, 0, 1, 0.32, 0.34, 0.13, 0.54]]

    test_features = [[0, 1, 0, 1, 0.11, 0.41, 0.11, 2.63],
                     [0, 0, 0, 1, 3.23, 1.45, 0.01, 1.22]]

    train_results = [0.5, 0.3, 0.9, 1.3]
    test_results = [1.5, 0.5]

    binary_feature_indices = [0, 1, 2, 3]

    def testPValueWorksAsIntended(self):
        model = self.trainModelWithExplicitNumberOfPhrases(10, True)

        for enet_model in model.models_by_phrase:  # fake the scores so that we don't have models which tie
            enet_model.score = random.random()

        score_0 = model.score(self.test_features, self.test_results)
        score_0_redundant = model.score(self.test_features, self.test_results)
        assert score_0 == score_0_redundant

        model.p = 0.5
        score_half = model.score(self.test_features, self.test_results)
        assert score_0 != score_half

        model.p = 1.0
        score_1 = model.score(self.test_features, self.test_results)
        assert score_0 != score_1
        assert score_half != score_1

    def testExplicitModelCountWorks(self):
        model = self.trainModelWithExplicitNumberOfPhrases(5, False)
        assert len(model.models_by_phrase) == 5

    def testDuplicatePhrasesAreNotCreated(self):
        model = self.trainModelWithExplicitNumberOfPhrases(5, False)
        assert len(model.models_by_phrase) == 5

        first_phrase = copy.deepcopy(model.models_by_phrase[0].phrase)
        assert first_phrase.equals(model.models_by_phrase[0].phrase)
        assert model.currentPhraseExists(first_phrase)

        first_phrase.is_or = not first_phrase.is_or
        assert not first_phrase.equals(model.models_by_phrase[0].phrase)
        assert not model.currentPhraseExists(first_phrase)

    def trainModelWithExplicitNumberOfPhrases(self, phrase_count, at_least):
        num_phrases = 0
        model = None
        explicit_count = 0
        if not at_least:
            explicit_count = phrase_count
        while (not at_least and num_phrases != phrase_count) or (at_least and num_phrases < phrase_count):

            model = RandomSubsetElasticNet(1, 0.5, self.binary_feature_indices, upper_bound=0.5, lower_bound=0, p=0,
                                           explicit_model_count=(explicit_count - 1))
            model.fit(self.train_features, self.train_results)
            num_phrases = len(model.models_by_phrase)
            [self.assertScore(model_phrase) for model_phrase in model.models_by_phrase if model_phrase.phrase.value is not None]

        return model

    def assertScore(self, phrase):
        assert phrase.score > 0

    def testParameterValidationWorks(self):
        bad_explicit_phrases = [RecursiveBooleanPhrase(5, 1, False, None)]
        self.assertInvalidParams([-1, 0, 1])
        self.assertInvalidParams([0, 1, "test"])
        self.assertInvalidParams(self.binary_feature_indices, alpha=-1)
        self.assertInvalidParams(self.binary_feature_indices, l_one_ratio=-1)
        self.assertInvalidParams(self.binary_feature_indices, upper_bound=5)
        self.assertInvalidParams(self.binary_feature_indices, lower_bound=-1)
        self.assertInvalidParams(self.binary_feature_indices, lower_bound=.3, upper_bound=.1)
        self.assertInvalidParams(self.binary_feature_indices, p=100)
        self.assertInvalidParams(self.binary_feature_indices, explicit_model_count=-2)
        self.assertInvalidParams(self.binary_feature_indices, max_boolean_generation_attempts=0)
        self.assertInvalidParams(self.binary_feature_indices, default_coverage_threshold=1.4)
        self.assertInvalidParams(self.binary_feature_indices, explicit_phrases=bad_explicit_phrases)

    def assertInvalidParams(self, binary_feature_indices, alpha=1, l_one_ratio=2, upper_bound=0.5, lower_bound=0.1, p=0,
                            explicit_model_count=-1, max_boolean_generation_attempts=10,
                            default_coverage_threshold=0.8, explicit_phrases=None):
        error = ""
        try:
            RandomSubsetElasticNet(alpha, l_one_ratio, binary_feature_indices, upper_bound=upper_bound,
                                   lower_bound=lower_bound, p=p, explicit_model_count=explicit_model_count,
                                   max_boolean_generation_attempts=max_boolean_generation_attempts,
                                   coverage_threshold=default_coverage_threshold, explicit_phrases=explicit_phrases)
        except AttributeError as attributeError:
            error = SafeCastUtil.safeCast(attributeError, str)
        assert "invalid parameters" in error

    def testRSENFailsIfNonBinaryMatrixSentIn(self):
        self.train_features[0][0] = 2
        error = ""
        try:
            model = RandomSubsetElasticNet(1, 2, self.binary_feature_indices)
            model.fit(self.train_features, self.train_results)
        except ValueError as valueError:
            error = SafeCastUtil.safeCast(valueError, str)
        assert "Non-binary feature" in error
