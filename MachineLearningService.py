import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy

from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil
from SupportedAnalysisTypes import SupportedAnalysisTypes


class MachineLearningService(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def __init__(self, data):
        self.data = data
        self.genomes_array = None
        self.third_party_response = None

    def analyze(self):
        m_val = 1
        max_depth = len(self.data.get(ArgumentProcessingService.RESULTS))
        features = self.extractFeatures()
        results = self.extractResults()
        analysis_type = SupportedAnalysisTypes.REGRESSION
        if self.data.get(ArgumentProcessingService.IS_CLASSIFIER):
            analysis_type = SupportedAnalysisTypes.CLASSIFICATION
        model = self.trainRandomForest(results, features, m_val, max_depth, analysis_type)
        self.log.info("Random Forest Model trained: %s", model.feature_importances_)
        return model

    def extractFeatures(self):
        features = []
        feature_map = self.data.get(ArgumentProcessingService.FEATURES)
        for key in feature_map.keys():
            if key is ArgumentProcessingService.FEATURE_NAMES:
                continue
            features.append(feature_map[key])
        return features

    def extractResults(self):
        results = []
        for cell_line in self.data.get(ArgumentProcessingService.RESULTS):
            results.append(cell_line[1])
        return results

    def trainRandomForest(self, results, features, m_val, max_depth, analysis_type):
        max_leaf_nodes = numpy.maximum(2, SafeCastUtil.safeCast(numpy.ceil(max_depth), int))
        max_features = SafeCastUtil.safeCast(numpy.floor(m_val), int)
        if analysis_type == SupportedAnalysisTypes.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_features=max_features)
        else:
            model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=max_leaf_nodes, max_features=max_features)
        model.fit(features, results)
        self.log.debug("Successful creation of Random Forest model: %s\n", model)
        return model
