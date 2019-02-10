from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from SupportedMachineLearningAlgorithms import SupportedMachineLearningAlgorithms
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.GeneListComboUtility import GeneListComboUtility
import os
import logging
import copy
import csv
import numpy

from Utilities.SafeCastUtil import SafeCastUtil


class RecommendationsService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    def __init__(self, inputs):
        self.inputs = inputs

    def recommendByHoldout(self, input_folder):
        # See self.inputs.get(ArgumentProcessingService.FEATURES). This contains a map representing all cell lines and
        # their features, along with a set of "featureNames".

        combos = self.determineGeneListCombos()

        # A dictionary of cell lines to their features, with the feature names also in there as one of the keys.
        # Both the features and the feature names are presented as an ordered list, all of them have the same length.
        cell_line_map = self.inputs.get(ArgumentProcessingService.FEATURES)
        # get results map
        for cell_line in cell_line_map.keys():
            if cell_line == ArgumentProcessingService.FEATURE_NAMES:
                continue
            trimmed_cell_lines = self.removeFromCellLinesMap(cell_line, cell_line_map)
            # remove cell line from results
            recs = []
            for drug in self.getDrugFolders(input_folder):
                best_model = self.determineAndTrainBestModel(drug, input_folder, trimmed_cell_lines, combos)
                if best_model is None:
                    continue
                # recs.append(best_model.predict(cell_line_map[cell_line]))

           # See which drug prediction comes closest to actual R^2 score.
           # See self.inputs.get(ArgumentProcessingService.RESULTS) for this value.
           # Record to FinalResults.csv
        pass

    def determineGeneListCombos(self):
        gene_lists = self.inputs.get(ArgumentProcessingService.GENE_LISTS)
        feature_names = self.inputs.get(ArgumentProcessingService.FEATURES).get(ArgumentProcessingService.FEATURE_NAMES)

        combos, expected_length = GeneListComboUtility.determineGeneListCombos(gene_lists, feature_names)

        if len(combos) != expected_length:
            self.log.warning("Unexpected number of combos detected, should be %s but instead created %s.\n%s",
                             expected_length, len(combos), combos)
        return combos


        # Create a dict for gene list combos and a dict for hyperparameters, where the keys are EN, RF and SVM.
        # In George's repository the analysis_files_folder is Repository/Runs/Analysis_files/
        # Find the best scoring gene list combos
        # genelists = collections.defaultdict(dict)
        # models = ['ElasticNetAnalysis', 'RandomForestAnalysis', 'RadialBasisFunctionSVMAnalysis']
        # for model in models:
        #     analysis = np.genfromtxt(analysis_files_folder + '/' + drug + '_analysis/' + model + '.csv', delimiter=',',
        #                              dtype='str')
        #     analysis = analysis[1:, :]
        #     R2 = analysis[:, 1]
        #     best = analysis[np.argmax(R2)]
        #     for gl in best[0].split(' '):
        #         gene_list = gl.split(':')[1].replace('gene_list_', '')
        #         typestr = gl.split(':')[0]
        #         if 'gex' in typestr:
        #             if model == 'ElasticNetAnalysis':
        #                 genelists['EN']['gex'] = gene_list
        #             elif model == 'RandomForestAnalysis':
        #                 genelists['RF']['gex'] = gene_list
        #             elif model == 'RadialBasisFunctionSVMAnalysis':
        #                 genelists['SVM']['gex'] = gene_list
        #         if 'mut' in typestr:
        #             if model == 'ElasticNetAnalysis':
        #                 genelists['EN']['mut'] = gene_list
        #             elif model == 'RandomForestAnalysis':
        #                 genelists['RF']['mut'] = gene_list
        #             elif model == 'RadialBasisFunctionSVMAnalysis':
        #                 genelists['SVM']['mut'] = gene_list
        #         if 'cnum' in typestr:
        #             continue
        # for model in {'EN', 'RF', 'SVM'}:
        #     for type in {'gex', 'mut'}:
        #         if type not in genelists[model].keys():
        #             genelists[model][type] = 'nan'
        # Find best hyperparameters for the given gene list combos
        # First make for each algo a list with the hyperparameters that were optimal for a MC run with this algorithm
        # hyperparams = collections.defaultdict(dict)
        # hyperparam_MClist_EN = {}
        # hyperparam_MClist_RF = {}
        # hyperparam_MClist_SVM = {}
        # with open(analysis_files_folder + '/' + drug + '_analysis/' + drug + '_diagnostics.txt', 'r') as fr:
        #     EN = 0
        #     RF = 0
        #     SVM = 0
        #     hyperpars = ''
        #     for line in fr:
        #         if EN == 1:
        #             hyperpars = line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             EN = 2
        #         elif EN == 2:
        #             hyperpars += line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             if hyperpars in hyperparam_MClist_EN.keys():
        #                 hyperparam_MClist_EN[hyperpars] += 1
        #             else:
        #                 hyperparam_MClist_EN[hyperpars] = 1
        #             hyperpars = ''
        #             EN = 0
        #         if RF == 1:
        #             hyperpars = line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             RF = 2
        #         elif RF == 2:
        #             hyperpars += line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             if hyperpars in hyperparam_MClist_RF.keys():
        #                 hyperparam_MClist_RF[hyperpars] += 1
        #             else:
        #                 hyperparam_MClist_RF[hyperpars] = 1
        #             hyperpars = ''
        #             RF = 0
        #         if SVM == 1:
        #             hyperpars = line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             SVM = 2
        #         elif SVM == 2:
        #             hyperpars += line.rstrip().split(' = ')[0] + '_' + line.rstrip().split(' = ')[0]
        #             if hyperpars in hyperparam_MClist_SVM.keys():
        #                 hyperparam_MClist_SVM[hyperpars] += 1
        #             else:
        #                 hyperparam_MClist_SVM[hyperpars] = 1
        #             hyperpars = ''
        #             SVM = 0
        #         if line.rstrip()[:13] != 'INFO:Trainers':
        #             continue
        #         if 'ElasticNetAnalysis' in line:
        #             ln = line.rstrip().split(' ')
        #             genelist_gex = genelists[drug]['EN_gex']
        #             genelist_mut = genelists[drug]['EN_mut']
        #             if 'gex_hgnc:gene_list_' + genelist_gex in line and genelist_mut == 'nan' and 'mut' not in line and 'cnum' not in line:
        #                 EN = 1
        #             elif 'mut_hgnc:gene_list_' + genelist_mut in line and genelist_gex == 'nan' and 'gex' not in line and 'cnum' not in line:
        #                 EN = 1
        #             elif 'gex_hgnc:gene_list_' + genelist_gex in line and 'mut_hgnc:gene_list_' + genelist_mut in line and 'cnum' not in line:
        #                 EN = 1
        #         elif 'RandomForestAnalysis' in line:
        #             ln = line.rstrip().split(' ')
        #             genelist_gex = genelists[drug]['RF_gex']
        #             genelist_mut = genelists[drug]['RF_mut']
        #             if 'gex_hgnc:gene_list_' + genelist_gex in line and genelist_mut == 'nan' and 'mut' not in line and 'cnum' not in line:
        #                 RF = 1
        #             elif 'mut_hgnc:gene_list_' + genelist_mut in line and genelist_gex == 'nan' and 'gex' not in line and 'cnum' not in line:
        #                 RF = 1
        #             elif 'gex_hgnc:gene_list_' + genelist_gex in line and 'mut_hgnc:gene_list_' + genelist_mut in line and 'cnum' not in line:
        #                 RF = 1
        #         elif 'RadialBasisFunctionSVMAnalysis' in line:
        #             ln = line.rstrip().split(' ')
        #             genelist_gex = genelists[drug]['SVM_gex']
        #             genelist_mut = genelists[drug]['SVM_mut']
        #             if 'gex_hgnc:gene_list_' + genelist_gex in line and genelist_mut == 'nan' and 'mut' not in line and 'cnum' not in line:
        #                 SVM = 1
        #             elif 'mut_hgnc:gene_list_' + genelist_mut in line and genelist_gex == 'nan' and 'gex' not in line and 'cnum' not in line:
        #                 SVM = 1
        #             elif 'gex_hgnc:gene_list_' + genelist_gex in line and 'mut_hgnc:gene_list_' + genelist_mut in line and 'cnum' not in line:
        #                 SVM = 1
        # Given the lists of hyperparameters selected in the MC runs, find the mode.
        # max = 0
        # argmax = ''
        # for k, v in hyperparam_MClist_EN.items():
        #     if v > max:
        #         max = v
        #         argmax = k
        # hyperparams['EN'][argmax.split('_')[0]] = argmax.split('_')[1]
        # hyperparams['EN'][argmax.split('_')[2]] = argmax.split('_')[3]
        # max = 0
        # argmax = ''
        # for k, v in hyperparam_MClist_RF.items():
        #     if v > max:
        #         max = v
        #         argmax = k
        # hyperparams['RF'][argmax.split('_')[0]] = argmax.split('_')[1]
        # hyperparams['RF'][argmax.split('_')[2]] = argmax.split('_')[3]
        # max = 0
        # argmax = ''
        # for k, v in hyperparam_MClist_SVM.items():
        #     if v > max:
        #         max = v
        #         argmax = k
        # hyperparams['SVM'][argmax.split('_')[0]] = argmax.split('_')[1]
        # hyperparams['SVM'][argmax.split('_')[2]] = argmax.split('_')[3]
        # return genelists, hyperparams

    def removeFromCellLinesMap(self, cell_line, features_map):
        cloned_features = copy.deepcopy(features_map)
        del cloned_features[ArgumentProcessingService.FEATURE_NAMES]
        del cloned_features[cell_line]
        return cloned_features

    def getDrugFolders(self, input_folder):
        # search for and return all drug folders in the input_folder.
        folders = os.listdir(input_folder)
        # TODO: Figure out required phrase to mark it as a drug folder
        drug_folders = [f for f in folders if 'Analysis' in f]
        return drug_folders

    def determineAndTrainBestModel(self, drug, analysis_files_folder, trimmed_cell_lines, combos):
        # TODO: ultimately we'd want to use multiple algorithms, and make an ensemble prediction/prescription.
        # But for now, let's stick with one algorithm.
        best_scoring_combo = None
        best_scoring_algo = None
        optimal_hyperparams = None
        top_score = AbstractModelTrainer.DEFAULT_MIN_SCORE
        for analysis_file_name in self.fetchAnalysisFiles(drug, analysis_files_folder):
            file = analysis_files_folder + "/" + drug + "/" + analysis_file_name
            with open(file, 'rt') as analysis_file:
                reader = csv.reader(analysis_file)
                try:
                    header = []
                    indices_of_outer_loops = []
                    line_index = -1
                    for row in reader:
                        line_index += 1
                        if line_index == 0:
                            header = row
                            for i in range(0, len(row)):
                                if MachineLearningService.SCORE_AND_HYPERPARAM_PHRASE in row[i]:
                                    indices_of_outer_loops.append(i)
                            continue
                        string_combo = row[header.index(MachineLearningService.FEATURE_FILE_GENE_LIST_COMBO)]
                        score = SafeCastUtil.safeCast(row[header.index(self.scorePhrase())], float)
                        if score is not None and score > top_score:
                            best_scoring_algo = analysis_file_name.split(".")[0]
                            best_scoring_combo = string_combo
                            top_score = score
                            optimal_hyperparams = self.fetchBestHyperparams(row, indices_of_outer_loops)
                except ValueError as valueError:
                    self.log.error(valueError)
                finally:
                    self.log.debug("Closing file %s", analysis_file)
                    analysis_file.close()
        if top_score <= 0:
            # TODO - Consider writing this to an explicit diagnostic file via extracting to first class service,
            # not just the process error log.
            self.log.error('Error: no method found an R2 higher than 0 for drug: %s.', drug)
            return None
        return self.trainBestModel(best_scoring_algo, best_scoring_combo, optimal_hyperparams, combos)

    def scorePhrase(self):
        if self.inputs.get(ArgumentProcessingService.IS_CLASSIFIER):
            return MachineLearningService.PERCENT_ACCURATE_PREDICTIONS
        return MachineLearningService.R_SQUARED_SCORE

    def fetchBestHyperparams(self, row, indices_of_outer_loops):
        monte_carlo_results = self.getMonteCarloResults(row, indices_of_outer_loops)
        best_hyps = None
        top_score = AbstractModelTrainer.DEFAULT_MIN_SCORE
        for hyperparam in SafeCastUtil.safeCast(monte_carlo_results.keys(), list):
            average_of_scores = numpy.average(monte_carlo_results.get(hyperparam))
            if average_of_scores > top_score:
                top_score = average_of_scores
                best_hyps = hyperparam
        return best_hyps

    def getMonteCarloResults(self, row, indices_of_outer_loops):
        hyperparams_to_scores = {}
        for i in range(0, len(row)):
            if i in indices_of_outer_loops:
                score_and_hyperparam = row[i].split(MachineLearningService.DELIMITER)
                score = SafeCastUtil.safeCast(score_and_hyperparam[0], float)
                if hyperparams_to_scores.get(score_and_hyperparam[1]) is not None:
                    hyperparams_to_scores[score_and_hyperparam[1]].append(score)
                else:
                    hyperparams_to_scores[score_and_hyperparam[1]] = [score]
        return hyperparams_to_scores

    def fetchAnalysisFiles(self, drug, input_folder):
        # return all "...Analysis.csv" files in path of input_folder/drug.
        files = os.listdir(input_folder + "/" + drug)
        return [file for file in files if "Analysis.csv" in file]

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
        # best = numpy.argmax(celline_viabilities[:, 1])
        # bestdrug = celline_viabilities[best, 0]
        # bestviab = celline_viabilities[best, 1]
        # viab_threshold = viability_acceptance * bestviab
        # prescription = [bestdrug]
        # for d in range(len(celline_viabilities[:, 1])):
        #     if celline_viabilities[d, 1] >= viab_threshold and celline_viabilities[d, 0] not in prescription:
        #         prescription.append(celline_viabilities[d, 0])
        # return prescription
        pass
