import inspect


class SupportedMachineLearningAlgorithms:

    RANDOM_FOREST = "RandomForestAnalysis"
    LINEAR_SVM = "LinearSVMAnalysis"
    RADIAL_BASIS_FUNCTION_SVM = "RadialBasisFunctionSVMAnalysis"
    ELASTIC_NET = "ElasticNetAnalysis"
    RIDGE_REGRESSION = "RidgeRegressionAnalysis"
    LASSO_REGRESSION = "LassoRegressionAnalysis"
    RANDOM_SUBSET_ELASTIC_NET = "RandomSubsetElasticNetAnalysis"

    @staticmethod
    def fetchAlgorithms():
        class_members = inspect.getmembers(SupportedMachineLearningAlgorithms, lambda a: not (inspect.isroutine(a)))
        return [algorithm[1] for algorithm in class_members if type(algorithm[1]) is str]
