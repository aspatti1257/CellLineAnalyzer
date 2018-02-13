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
        print ("data", self.data)
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

    def hyperparameterTuning (self, dataframe):
        param_lst = {"n_estimators": range(30, 50)}
        num_trials_outer = 2
        num_trials_inner = 2
        r2_rf = []
        for outerMCCV in range(num_trials_outer):
            out_x_train, out_x_test, out_y_train, out_y_test = train_test_split(dataframe, auc, test_size=0.2, random_state=42)
            out_y_train = out_y_train.flatten()
            out_y_test = out_y_test.flatten()
            param_outer, score_outer, param_inner, score_inner = [], [], [], []
            for innerMCCV in range(num_trials_inner):
                in_x_train, in_x_test, in_y_train, in_y_test = train_test_split(out_x_train, out_y_train, test_size=0.2, random_state=42)
                clf = RandomForestRegressor()
                grid = GridSearchCV(clf, param_grid=param_lst, cv=2)
                grid.fit(in_x_train, in_y_train)
                results = grid.cv_results_
                best_fit = np.argmax(results.get("mean_test_score"))
                r2 = results.get("mean_test_score")
                get_params = results.get("params")[best_fit]
                param_inner.append([get_params])
                score_inner.append([r2])
                print(r2, get_params)
            score_outer = list(map(lambda x: np.mean(x), score_inner))
            best_case = np.argmax(score_outer)
            param_outer = list(map(lambda x: x, param_inner[best_case]))
            print(param_outer)
            clf = RandomForestRegressor(n_estimators=param_outer[0]["n_estimators"])
            clf.fit(out_x_train, out_y_train)
            r2 = clf.score(out_x_test, out_y_test)
            r2_rf.append(r2)
        return np.average(r2_rf)