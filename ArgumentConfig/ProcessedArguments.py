from ArgumentConfig.AnalysisType import AnalysisType


class ProcessedArguments(object):

    def __init__(self, results, is_classifier, features, gene_lists, inner_monte_carlo_permutations,
                 outer_monte_carlo_permutations, data_split, algorithm_configs, num_threads, record_diagnostics,
                 individual_train_config, rsen_config, recs_config, specific_combos, analyze_all):
        self.results = results
        self.is_classifier = is_classifier
        self.features = features
        self.gene_lists = gene_lists
        self.inner_monte_carlo_permutations = inner_monte_carlo_permutations
        self.outer_monte_carlo_permutations = outer_monte_carlo_permutations
        self.data_split = data_split
        self.algorithm_configs = algorithm_configs
        self.num_threads = num_threads
        self.record_diagnostics = record_diagnostics
        self.individual_train_config = individual_train_config
        self.rsen_config = rsen_config
        self.recs_config = recs_config
        self.specific_combos = specific_combos
        self.analyze_all = analyze_all

    def analysisType(self):
        if self.recs_config is not None and self.recs_config.viability_acceptance is not None:
            return AnalysisType.RECOMMENDATIONS
        if self.analyze_all:
            return AnalysisType.NO_GENE_LISTS
        elif len(self.specific_combos) > 0:
            return AnalysisType.FULL_CLA_SPECIFIC_COMBO
        elif self.individual_train_config.shouldTrainIndividualCombo():
            return AnalysisType.INDIVIDUAL_TRAIN
        else:
            return AnalysisType.FULL_CLA
