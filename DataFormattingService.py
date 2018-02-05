import ...
from sklearn import preprocessing

class DataFormattingService(object):

    def __init__(self, inputs):
        self.inputs = inputs

    def formatData(self):
        #  TODO - Christine: Handle one-hot encoding and hyperparameter optimization.

        def encode_categorical(array):
            if not array.dtype == np.dtype('float64'):
                return preprocessing.LabelEncoder().fit_transform(array)
            else:
                return array

        # Encode sites as categorical variables
        def one_hot (dataframe):

            # Categorical columns for use in one-hot encoder
            categorical = (dataframe.dtypes.values != np.dtype('float64'))

            # Encode all labels
            dataframe = dataframe.apply(encode_categorical)
            dataframe_np = dataframe.as_matrix()
            return dataframe_np, dataframe

        # Hyperparameter tuning
        param_lst = {"rf": {"n_estimators": range(10, 30)}}
        algo_lst = ["rf"]

        def tuning (dataframe):
            num_trials_outer = 10
            num_trials_inner = 10
            r2_rf = []
            for outerMCCV in range(num_trials_outer):
                out_x_train, out_x_test, out_y_train, out_y_test = train_test_split(dataframe, auc, test_size=0.2, random_state=42)
                out_y_train = out_y_train.flatten()
                out_y_test = out_y_test.flatten()
                param_outer, score_outer = {"rf": []}, {"rf": []}
                param_inner, score_inner = {"rf": []}, {"rf": []}
                for innerMCCV in range(num_trials_inner):
                    in_x_train, in_x_test, in_y_train, in_y_test = train_test_split(out_x_train, out_y_train, test_size=0.2, random_state=42)
                    for algo in algo_lst:
                        clf = RandomForestRegressor()
                        grid = GridSearchCV(clf, param_grid=param_lst[algo], cv=5)
                        grid.fit(in_x_train, in_y_train)
                        results = grid.cv_results_
                        best_fit = np.argmax(results.get("mean_test_score"))
                        r2 = results.get("mean_test_score")
                        get_params = results.get("params")[best_fit]
                        param_inner[algo].append([get_params])
                        score_inner[algo].append([r2])
                        print (algo, r2)
                for algo in algo_lst:
                    score_outer[algo] = list(map(lambda x: np.mean(x), score_inner[algo]))
                print (score_outer)
                for algo in algo_lst:
                    best_case = np.argmax(score_outer[algo])
                    param_outer[algo] = list(map(lambda x: x, param_inner[algo][best_case]))
                    print (param_outer)
                    clf = RandomForestRegressor(n_estimators=param_outer[algo][0]["n_estimators"])
                    clf.fit(out_x_train, out_y_train)
                    r2 = clf.score(out_x_test, out_y_test)
                    if algo == "rf":
                        r2_rf.append(r2)
            return np.average(r2_rf)