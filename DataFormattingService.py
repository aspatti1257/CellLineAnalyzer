import numpy as np
import pandas as pd
from sklearn import preprocessing
import logging

class DataFormattingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self):
        pass
        #  TODO - Christine: Handle one-hot encoding and hyperparameter optimization.

    def encode_categorical(self, array):
        if not array.dtype == np.dtype('float64'):
            return preprocessing.LabelEncoder().fit_transform(array)
        else:
            return array

    # Encode sites as categorical variables
    def one_hot (self, dataframe):

        # Categorical columns for use in one-hot encoder
        categorical = (dataframe.dtypes.values != np.dtype('float64'))

        # Encode all labels
        dataframe = dataframe.apply(self.encode_categorical)
        dataframe_np = dataframe.as_matrix()
        # assert isinstance(dataframe, object)
        return dataframe_np, dataframe


    # Binary one hot encoding
    def binary_one_hot(self, dataframe):

        dataframe_binary_pd = pd.get_dummies(dataframe)
        dataframe_binary = dataframe_binary_pd.as_matrix()
        return dataframe_binary, dataframe_binary_pd

    # Hyperparameter tuning
    param_lst = {"n_estimators": range(10, 30)}

    def tuning (self, dataframe):
        num_trials_outer = 2
        num_trials_inner = 2
        r2_rf = []
        for outerMCCV in range(num_trials_outer):
            out_x_train, out_x_test, out_y_train, out_y_test = train_test_split(exp_cn, auc, test_size=0.2,
                                                                                random_state=42)
            out_y_train = out_y_train.flatten()
            out_y_test = out_y_test.flatten()
            param_outer, score_outer, param_inner, score_inner = [], [], [], []
            for innerMCCV in range(num_trials_inner):
                in_x_train, in_x_test, in_y_train, in_y_test = train_test_split(out_x_train, out_y_train, test_size=0.2,
                                                                                random_state=42)
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
        print("rf :", np.average(r2_rf))

s = DataFormattingService(object)
categorical_pd = pd.read_csv('Testing/SampleClassifierDataFolder/categorical.csv', delimiter=',')
categorical = np.genfromtxt('Testing/SampleClassifierDataFolder/categorical.csv', delimiter=',')
print(s.binary_one_hot(categorical_pd))
