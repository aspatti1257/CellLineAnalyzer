from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.ElasticNetTrainer import ElasticNetTrainer
from Trainers.LassoRegressionTrainer import LassoRegressionTrainer
from Trainers.LinearSVMTrainer import LinearSVMTrainer
from Trainers.RadialBasisFunctionSVMTrainer import RadialBasisFunctionSVMTrainer
from Trainers.RandomForestTrainer import RandomForestTrainer
from Trainers.RandomSubsetElasticNetTrainer import RandomSubsetElasticNetTrainer
from Trainers.RidgeRegressionTrainer import RidgeRegressionTrainer


class ModelTrainerFactory(object):

    @staticmethod
    def createTrainerFromTargetAlgorithm(is_classifier, target_algorithm, rsen_config):
        if target_algorithm == SupportedMachineLearningAlgorithms.RANDOM_FOREST:
            trainer = RandomForestTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.LINEAR_SVM:
            trainer = LinearSVMTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.RADIAL_BASIS_FUNCTION_SVM:
            trainer = RadialBasisFunctionSVMTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.ELASTIC_NET and not is_classifier:
            trainer = ElasticNetTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.RIDGE_REGRESSION and not is_classifier:
            trainer = RidgeRegressionTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.LASSO_REGRESSION and not is_classifier:
            trainer = LassoRegressionTrainer(is_classifier)
        elif target_algorithm == SupportedMachineLearningAlgorithms.RANDOM_SUBSET_ELASTIC_NET and \
                not is_classifier and rsen_config.binary_cat_matrix is not None and rsen_config.p_val is not None and \
                rsen_config.k_val is not None:
            trainer = RandomSubsetElasticNetTrainer(is_classifier, rsen_config.binary_cat_matrix, rsen_config.p_val,
                                                    rsen_config.k_val)
        else:
            raise ValueError("Unsupported Machine Learning algorithm: " + target_algorithm)
        return trainer
