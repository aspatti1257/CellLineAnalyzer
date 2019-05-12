from collections import OrderedDict

from ArgumentProcessingService import ArgumentProcessingService
from ArgumentConfig.AnalysisType import AnalysisType
from DataFormattingService import DataFormattingService
from MachineLearningService import MachineLearningService
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.GeneListComboUtility import GeneListComboUtility
import os
import logging
import copy
import csv
import numpy

from Utilities.ModelTrainerFactory import ModelTrainerFactory
from Utilities.SafeCastUtil import SafeCastUtil


class RecommendationsService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    def __init__(self, inputs):
        self.inputs = inputs

    def recommendByHoldout(self, input_folder):
        # TODO: Support for inputs to be a dict of drug_name => input, not just one set of inputs for all drugs.

        drug_to_cell_line_to_prediction_map = {}
        for drug in self.inputs.keys():
            drug_to_cell_line_to_prediction_map[drug] = {}
            processed_arguments = self.inputs.get(drug)
            combos = self.determineGeneListCombos(processed_arguments)

            # A dictionary of cell lines to their features, with the feature names also in there as one of the keys.
            # Both the features and the feature names are presented as an ordered list, all of them have the same length.
            cell_line_map = processed_arguments.features
            results = processed_arguments.results

            cloned_inputs = copy.deepcopy(processed_arguments)
            cloned_inputs.data_split = 1.0
            data_formatting_service = DataFormattingService(cloned_inputs)
            formatted_inputs = data_formatting_service.formatData(True, True)
            feature_names = formatted_inputs.get(ArgumentProcessingService.FEATURE_NAMES)
            # get results map
            for cell_line in cell_line_map.keys():
                if cell_line == ArgumentProcessingService.FEATURE_NAMES:
                    # Continue skips over this, so if the key we're analyzing isn't a cell line (i.e. it's the feature
                    # names, skip it).
                    continue
                trimmed_cell_lines, trimmed_results = self.removeCellLineFromFeaturesAndResults(cell_line, formatted_inputs,
                                                                                                results)
                # remove cell line from results
                cellline_viabilities = []
                best_model, best_combo = self.fetchBestModelAndCombo(drug, input_folder, trimmed_cell_lines,
                                                                     trimmed_results, combos, processed_arguments)
                if best_model is None or best_combo is None:
                    continue
                else:
                    prediction = self.generatePrediction(best_model, best_combo, cell_line, feature_names, formatted_inputs)
                    cellline_viabilities.append([drug, prediction])
                recs = []
                drug_to_cell_line_to_prediction_map[drug][cell_line] = prediction
                #self.presciption_from_prediction(self, trainer, viability_acceptance, cellline_viabilities)
                # @AP: viability_acceptance is a user-defined value, which should come from the arguments file. How do I
                # get it here?
                # @MB: Added it via referencing self.inputs.recs_config.viability_acceptance. Defaults to None, set to 0.1
                # for the sake of RecommendationsServiceIT testing.
                with open('FinalResults.csv','a') as f:
                    f.write(str(cell_line)+',')
                    for drug in recs:
                        f.write(drug+';')
                    f.write('\n')
               # See which drug prediction comes closest to actual R^2 score.
               # See self.inputs.results for this value.
               # Record to FinalResults.csv
            pass

    def determineGeneListCombos(self, processed_arguments):
        gene_lists = processed_arguments.gene_lists
        feature_names = processed_arguments.features.get(ArgumentProcessingService.FEATURE_NAMES)

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

    def removeCellLineFromFeaturesAndResults(self, cell_line, formatted_inputs, results):
        cloned_formatted_data = copy.deepcopy(formatted_inputs)
        del cloned_formatted_data.get(DataFormattingService.TRAINING_MATRIX)[cell_line]

        cloned_results = [result for result in results if result[0] is not cell_line]
        return cloned_formatted_data, cloned_results

    def getDrugFolders(self, input_folder):
        # search for and return all drug folders in the input_folder.
        folders = os.listdir(input_folder)
        # TODO: Figure out required phrase to mark it as a drug folder
        drug_folders = [f for f in folders if 'Analysis' in f]
        return drug_folders

    def fetchBestModelAndCombo(self, drug, analysis_files_folder, trimmed_cell_lines, trimmed_results, combos,
                               processed_arguments):
        # TODO: ultimately we'd want to use multiple algorithms, and make an ensemble prediction/prescription.
        # But for now, let's stick with one algorithm.
        best_combo_string = None
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
                        score = SafeCastUtil.safeCast(row[header.index(self.scorePhrase(processed_arguments))], float)
                        if score is not None and score > top_score:
                            best_scoring_algo = analysis_file_name.split(".")[0]
                            best_combo_string = string_combo
                            top_score = score
                            optimal_hyperparams = self.fetchBestHyperparams(row, indices_of_outer_loops)
                except ValueError as valueError:
                    #@AP what does this do?
                    #@MB: This is a try/catch block. Any errors that we encounter will be caught and handled here. In
                    # this case, we're just logging it, but we may want to do something else in the future if we
                    # discover something really bad. This helps us quickly detect when something goes wrong.
                    self.log.error(valueError)
                finally:
                    self.log.debug("Closing file %s", analysis_file)
                    analysis_file.close()
        if top_score <= 0:
            # TODO - Consider writing this to an explicit diagnostic file via extracting to first class service,
            # not just the process error log.
            self.log.error('Error: no method found an R2 higher than 0 for drug: %s.', drug)
            return None

        best_combo = self.determineBestComboFromString(best_combo_string, combos, processed_arguments)
        return self.trainBestModelWithCombo(best_scoring_algo, best_combo, optimal_hyperparams, trimmed_cell_lines,
                                            trimmed_results, processed_arguments)


    def scorePhrase(self, processed_arguments):
        if processed_arguments.is_classifier:
            return MachineLearningService.PERCENT_ACCURATE_PREDICTIONS
        return MachineLearningService.R_SQUARED_SCORE

    def fetchBestHyperparams(self, row, indices_of_outer_loops):
        monte_carlo_results = self.getMonteCarloResults(row, indices_of_outer_loops)
        best_hyps = None
        top_score = AbstractModelTrainer.DEFAULT_MIN_SCORE
        max_num_occurrences = 0
        best_hyps_list = []
        for hyperparam in SafeCastUtil.safeCast(monte_carlo_results.keys(), list):
            if len(monte_carlo_results.get(hyperparam)) > max_num_occurrences:
                max_num_occurrences = len(monte_carlo_results.get(hyperparam))
                best_hyps_list = [hyperparam]
            elif len(monte_carlo_results.get(hyperparam)) == max_num_occurrences:
                best_hyps_list.append(hyperparam)
        if len(best_hyps_list) == 1:
            best_hyps = hyperparam
            top_score = numpy.average(monte_carlo_results.get(hyperparam))
        elif len(best_hyps_list) > 1:
           top_score = 0
           for hyperparam in best_hyps_list:
               if numpy.average(monte_carlo_results.get(hyperparam)) > top_score:
                   top_score = numpy.average(monte_carlo_results.get(hyperparam))
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

    def trainBestModelWithCombo(self, best_scoring_algo, best_scoring_combo, optimal_hyperparams, trimmed_cell_lines,
                                trimmed_results, processed_arguments):
        is_classifier = processed_arguments.is_classifier
        rsen_config = processed_arguments.rsen_config
        training_matrix = GeneListComboUtility.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX,
                                                                      best_scoring_combo, trimmed_cell_lines,
                                                                      AnalysisType.RECOMMENDATIONS)
        trainer = ModelTrainerFactory.createTrainerFromTargetAlgorithm(is_classifier, best_scoring_algo, rsen_config)

        features, relevant_results = trainer.populateFeaturesAndResultsByCellLine(training_matrix, trimmed_results)
        params = [SafeCastUtil.safeCast(param.split(":")[1].strip(), float) for param in optimal_hyperparams.split(",")]
        feature_names = training_matrix.get(ArgumentProcessingService.FEATURE_NAMES)
        model = trainer.buildModel(relevant_results, features, params, feature_names)
        return model, best_scoring_combo

        # @AP: feature_names is an input parameter whenever the function is called in MachineLearningService. However,
        #  when I look into the code for the trainers, it is never used. Is it obsolete?
        # @MB: No, it's not obselete. It's actually pretty important here. What we need to do is split the data into two
        # via DataFormattingService.
        # @AP Now we still need predictions. In MachineLearningService.py I see "trainer.fetchPredictionsAndScore", but
        #  I don't see this in for example the random forest trainer. Where can I find it? Am I overlooking something?
        # @MB: No, this isn't implemented for RandomForestTrainer because it's on the parent class AbstractModelTrainer.
        # This is an example of inheritance/polymorphism. So each RandomForestTrainer can also use that method because
        # every RandomForestTrainer is also an AbstractModelTrainer.

    def presciption_from_prediction(self, trainer, viability_acceptance, druglist, cellline_viabilities):
        # celline_viabilities has two columns: column 1 is a drugname, column 2 its (predicted) viability
        # viability_acceptance is a user-defined threshold: include all drugs whose performance
        # is >= viability_acceptance*best_viability
        # druglist is a lists the drugs for which viability of this cell line was predicted
        best = numpy.argmax(cellline_viabilities[:, 1])
        bestdrug = cellline_viabilities[best, 0]
        bestviab = cellline_viabilities[best, 1]
        viab_threshold = viability_acceptance * bestviab
        prescription = [bestdrug]
        for d in range(len(cellline_viabilities[:, 1])):
            if cellline_viabilities[d, 1] >= viab_threshold and cellline_viabilities[d, 0] not in prescription:
                prescription.append(cellline_viabilities[d, 0])
        return prescription

    def determineBestComboFromString(self, best_combo_string, combos, processed_arguments):
        gene_lists = processed_arguments.gene_lists
        combine_gene_lists = processed_arguments.rsen_config.combine_gene_lists
        analysis_type = processed_arguments.analysisType()
        for combo in combos:
            feature_set_string = GeneListComboUtility.generateFeatureSetString(combo, gene_lists,
                                                                               combine_gene_lists, analysis_type)
            if feature_set_string == best_combo_string:
                return combo
        return None

    def generatePrediction(self, best_model, best_combo, cell_line, all_features, formatted_inputs):
        ommited_cell_line = formatted_inputs.get(DataFormattingService.TRAINING_MATRIX).get(cell_line)
        input_wrapper = OrderedDict()
        input_wrapper[DataFormattingService.TRAINING_MATRIX] = OrderedDict()
        input_wrapper[DataFormattingService.TRAINING_MATRIX][cell_line] = ommited_cell_line
        input_wrapper[ArgumentProcessingService.FEATURE_NAMES] = all_features
        trimmed_cell_line = GeneListComboUtility.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX,
                                                                        best_combo, input_wrapper, AnalysisType.RECOMMENDATIONS)
        prediction = best_model.predict([trimmed_cell_line.get(cell_line)])[0]
        return prediction
