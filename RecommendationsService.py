from ArgumentProcessingService import ArgumentProcessingService
import numpy


class RecommendationsService(object):

    def __init__(self, inputs):
        self.inputs = inputs

    def recommendByHoldout(self, input_folder):
        # See self.inputs.get(ArgumentProcessingService.FEATURES). This contains a map representing all cell lines and
        # their features, along with a set of "featureNames".

        # combos = self.determineGeneListCombos()

        # cell_line_map = self.inputs.get(ArgumentProcessingService.FEATURES);
        # for cell_line in cell_line_map.keys():
        #    trimmed_cell_lines = self.removeFromCellLinesMap(cell_line, cell_line_map)
        #    recs = []
        #    for drug in self.getDrugFolders():
        #       best_model = determineAndTrainBestModel(drug, input_folder, trimmed_cell_lines, combos);
        #       recs.append(best_model.predict(cell_line_map[cell_line]))
        #
        #    See which drug prediction comes closest to actual R^2 score.
        #    See self.inputs.get(ArgumentProcessingService.RESULTS) for this value.
        #    Record to FinalResults.csv
        pass

    def determineGeneListCombos(self):
        # reference the function in MachineLearningService. Ideally remove it from that and put it in a new utility
        # class. e.g. GeneListComboUtility, which can handle generating the gene list combos. Then anywhere in the code
        # base which references this can go through this common utility.
        pass

    def removeFromCellLinesMap(self, cell_line, features_map):
        # return a clone of this object with the cell line removed from the dictionary.
        # Make sure to not edit the original. Make sure to skip the "featureNames" attribute.
        pass

    def getDrugFolders(self, input_folder):
        # search for and return all drug folders in the input_folder.
        pass

    def determineAndTrainBestModel(self, drug, input_folder, trimmed_cell_lines, combos):
        # best_scoring_algo = None
        # best_scoring_combo = None
        # optimal_hyperparams = None
        # top_score = 0
        # for analysis_file in self.fetchAnalysisFiles(drug, input_folder):
        #     for row in analysis_file: # need to open the .csv here and read them in.
        #         for monte_carlo_perm in row: # find the elements in the row reflecting outer monte carlo loops
        #             if monte_carlo_perm.score() > top_score:
        #                 top_score = monte_carlo_perm.score()
        #                 best_scoring_algo = analysis_file # trim the .csv part.
        #                 best_scoring_combo = monte_carlo_perm.combo()
        #                 optimal_hyperparams = monte_carlo_perm.hyperparams()

        # return trainBestModel(best_scoring_algo, best_scoring_combo, optimal_hyperparams, combos)
        pass

    def fetchAnalysisFiles(self, drug, input_folder):
        # return all "...Analysis.csv" files in path of input_folder/drug.
        pass

    def trainBestModel(self, best_scoring_algo, best_scoring_combo, optimal_hyperparams, combos):
        # for combo in combos:
        #    # maybe we want to extract this MachineLearningService function to GeneListComboUtility as well
        #    if MachineLearningService.generateFeatureSetString(combo) == best_scoring_combo:
        #       # create new trainer object for best_scoring_algo.
        #       # trim data for best_scoring_combo
        #       # train model with optimal_hyperparams and return it.
        # return None
        pass

    def presciption_from_prediction(self, viability_acceptance, druglist, celline_viabilities):
        # celline_viabilities has two columns: column 1 is a drugname, column 2 its (predicted) viability
        # viability_acceptance is a user-defined threshold: include all drugs whose performance
        # is >= viability_acceptance*best_viability
        # druglist is a lists the drugs for which viability of this cell line was predicted
        best = numpy.argmax(celline_viabilities[:, 1])
        bestdrug = celline_viabilities[best, 0]
        bestviab = celline_viabilities[best, 1]
        viab_threshold = viability_acceptance * bestviab
        prescription = [bestdrug]
        for d in range(len(celline_viabilities[:, 1])):
            if celline_viabilities[d, 1] >= viab_threshold and celline_viabilities[d, 0] not in prescription:
                prescription.append(celline_viabilities[d, 0])
        return prescription
