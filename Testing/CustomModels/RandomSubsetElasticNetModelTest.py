import unittest
import logging
import random


from CustomModels.RandomSubsetElasticNetModel import RandomSubsetElasticNetModel


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
        model = self.trainModelWithExplicitNumberOfPhrases(2, True)

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

    def trainModelWithExplicitNumberOfPhrases(self, phrase_count, at_least):
        num_phrases = 0
        model = None
        explicit_count = -1
        if not at_least:
            explicit_count = phrase_count
        while (not at_least and num_phrases != phrase_count) or (at_least and num_phrases < phrase_count):

            model = RandomSubsetElasticNetModel(1, 2, self.binary_feature_indices, upper_bound=0.5, lower_bound=0, p=0,
                                                explicit_model_count=(explicit_count - 1))
            model.fit(self.train_features, self.train_results)
            num_phrases = len(model.models_by_phrase)

        return model


