from collections import OrderedDict
import os
import copy
import csv
import numpy
import multiprocessing
import threading
from joblib import Parallel, delayed

from ArgumentProcessingService import ArgumentProcessingService
from ArgumentConfig.AnalysisType import AnalysisType
from DataFormattingService import DataFormattingService
from LoggerFactory import LoggerFactory
from MachineLearningService import MachineLearningService
from RecommendationsModelInfo import RecommendationsModelInfo
from Trainers.AbstractModelTrainer import AbstractModelTrainer
from Utilities.DictionaryUtility import DictionaryUtility
from Utilities.GeneListComboUtility import GeneListComboUtility
from Utilities.ModelTrainerFactory import ModelTrainerFactory
from Utilities.SafeCastUtil import SafeCastUtil


class RecommendationsService(object):

    log = LoggerFactory.createLog(__name__)

    PRE_REC_ANALYSIS_FILE = "PreRecAnalysis.csv"
    PREDICTIONS_FILE = "Predictions.csv"

    HEADER = "header"

    STD_DEVIATION = "std_deviation"
    MEAN = "mean"
    MEDIAN = "median"

    def __init__(self, inputs):
        self.inputs = inputs

    def analyzeRecommendations(self, input_folder):
        self.preRecsAnalysis(input_folder)
        self.recommendByHoldout(input_folder)

    def preRecsAnalysis(self, input_folder):
        self.log.info("Performing pre-recs analysis on all drugs.")
        drugs = self.inputs.keys()
        cell_line_predictions_by_drug = OrderedDict()
        header = numpy.concatenate((["cell_line"], SafeCastUtil.safeCast(drugs, list)), axis=0)
        cell_line_predictions_by_drug[self.HEADER] = header
        cell_line_predictions_by_drug[self.STD_DEVIATION] = [self.STD_DEVIATION]
        cell_line_predictions_by_drug[self.MEAN] = [self.MEAN]
        cell_line_predictions_by_drug[self.MEDIAN] = [self.MEDIAN]
        for drug in drugs:
            processed_arguments = self.inputs.get(drug)
            results = processed_arguments.results
            combos = self.determineGeneListCombos(processed_arguments)

            processed_arguments.data_split = 1.0
            data_formatting_service = DataFormattingService(processed_arguments)
            formatted_inputs = data_formatting_service.formatData(True, True)
            self.log.info("Determining best combo and score for drug %s.", drug)
            recs_model_info = self.fetchBestModelComboAndScore(drug, input_folder, formatted_inputs,
                                                               results, combos, processed_arguments)

            if recs_model_info is None or recs_model_info.model is None or recs_model_info.combo is None:
                continue
            self.generateMultiplePredictions(recs_model_info, formatted_inputs, results, cell_line_predictions_by_drug)

        self.writePreRecAnalysisFile(cell_line_predictions_by_drug, input_folder)

    def generateMultiplePredictions(self, recs_model_info, formatted_inputs, results, cell_line_predictions_by_drug):
        trimmed_matrix = GeneListComboUtility.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX,
                                                                     recs_model_info.combo, formatted_inputs,
                                                                     AnalysisType.RECOMMENDATIONS)

        features, relevant_results = recs_model_info.trainer.populateFeaturesAndResultsByCellLine(trimmed_matrix, results)
        cell_lines_in_order = [key for key in trimmed_matrix.keys() if key is not ArgumentProcessingService.FEATURE_NAMES]
        predictions = recs_model_info.model.predict(features)

        for i in range(0, len(cell_lines_in_order)):
            cell_line = cell_lines_in_order[i]
            if cell_line_predictions_by_drug.get(cell_line) is not None:
                cell_line_predictions_by_drug[cell_line].append(predictions[i])
            else:
                max_dict_length = 2
                for key in cell_line_predictions_by_drug.keys():
                    if key == self.HEADER:
                        continue
                    if len(cell_line_predictions_by_drug[key]) > max_dict_length:
                        max_dict_length = len(cell_line_predictions_by_drug[key])
                row = [cell_line]
                for _ in range(2, max_dict_length):
                    row.append(MachineLearningService.DELIMITER)
                row.append(predictions[i])
                cell_line_predictions_by_drug[cell_line] = row
        cell_line_predictions_by_drug[self.STD_DEVIATION].append(numpy.std(predictions))
        cell_line_predictions_by_drug[self.MEAN].append(numpy.mean(predictions))
        cell_line_predictions_by_drug[self.MEDIAN].append(numpy.median(predictions))

    def writePreRecAnalysisFile(self, cell_line_predictions_by_drug, input_folder):
        with open(input_folder + "/" + self.PRE_REC_ANALYSIS_FILE, "w", newline='') as pre_rec_analysis_file:
            try:
                writer = csv.writer(pre_rec_analysis_file)
                for key in cell_line_predictions_by_drug.keys():
                    writer.writerow(cell_line_predictions_by_drug.get(key))
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", pre_rec_analysis_file, error)
            finally:
                pre_rec_analysis_file.close()

    def recommendByHoldout(self, input_folder):
        # TODO: Support for inputs to be a dict of drug_name => input, not just one set of inputs for all drugs.
        self.log.info("Starting recommendation by holdout analysis on all drugs.")
        max_nodes = multiprocessing.cpu_count()

        for drug in self.inputs.keys():
            self.log.info("Starting recommendation by holdout analysis on specific drug %s.", drug)
            self.handleDrug(drug, input_folder, max_nodes, self.inputs.get(drug))

    def handleDrug(self, drug, input_folder, max_nodes, processed_arguments):
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

        requested_threads = processed_arguments.num_threads
        nodes_to_use = numpy.amin([requested_threads, max_nodes])

        Parallel(n_jobs=nodes_to_use)(delayed(self.handleCellLine)(cell_line, combos, drug, feature_names,
                                                                   formatted_inputs, input_folder,
                                                                   processed_arguments, results)
                                      for cell_line in cell_line_map.keys())

    def handleCellLine(self, cell_line, combos, drug, feature_names, formatted_inputs, input_folder,
                       processed_arguments, results):
        self.log.info("Holding out cell line %s for drug %s", cell_line, drug)
        if cell_line == ArgumentProcessingService.FEATURE_NAMES:\
            return
        trimmed_cell_lines, trimmed_results = self.removeNonNullCellLineFromFeaturesAndResults(cell_line,
                                                                                               formatted_inputs,
                                                                                               results)
        recs_model_info = self.fetchBestModelComboAndScore(drug, input_folder, trimmed_cell_lines,
                                                           trimmed_results, combos, processed_arguments)
        if recs_model_info is None or recs_model_info.model is None or recs_model_info.combo is None:
            self.log.warn("Unable to train best model or get best combo for cell line %s and drug %s.", cell_line, drug)
            return

        prediction = self.generateSinglePrediction(recs_model_info.model, recs_model_info.combo,
                                                   cell_line, feature_names, formatted_inputs)

        self.writeToPredictionsCsvInLock(cell_line, drug, input_folder, prediction, recs_model_info.score)

    def writeToPredictionsCsvInLock(self, cell_line, drug, input_folder, prediction, score):
        self.log.debug("Locking current thread %s.", threading.current_thread())
        lock = threading.Lock()
        lock.acquire(True)
        write_action = "w"
        if self.PREDICTIONS_FILE in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + self.PREDICTIONS_FILE, write_action, newline='') as predictions_file:
            try:
                writer = csv.writer(predictions_file)
                if write_action == "w":
                    writer.writerow(["Drug", "Cell_Line", "Prediction", "R2^Score"])
                line = [drug, cell_line, SafeCastUtil.safeCast(prediction, str), SafeCastUtil.safeCast(score, str)]
                writer.writerow(line)
            except ValueError as error:
                self.log.error("Error writing to file %s. %s", self.PREDICTIONS_FILE, error)
            finally:
                predictions_file.close()
                self.log.debug("Releasing current thread %s.", threading.current_thread())
                lock.release()

    def determineGeneListCombos(self, processed_arguments):
        gene_lists = processed_arguments.gene_lists
        feature_names = processed_arguments.features.get(ArgumentProcessingService.FEATURE_NAMES)

        combos, expected_length = GeneListComboUtility.determineGeneListCombos(gene_lists, feature_names)

        if len(combos) != expected_length:
            self.log.warning("Unexpected number of combos detected, should be %s but instead created %s.\n%s",
                             expected_length, len(combos), combos)
        return combos

    def removeNonNullCellLineFromFeaturesAndResults(self, cell_line, formatted_inputs, results):
        cloned_formatted_data = copy.deepcopy(formatted_inputs)
        if cell_line is not None:
            del cloned_formatted_data.get(DataFormattingService.TRAINING_MATRIX)[cell_line]

        cloned_results = [result for result in results if result[0] is not cell_line and cell_line is not None]
        return cloned_formatted_data, cloned_results

    def getDrugFolders(self, input_folder):
        folders = os.listdir(input_folder)
        # TODO: Figure out required phrase to mark it as a drug folder
        drug_folders = [f for f in folders if 'Analysis' in f]
        return drug_folders

    def fetchBestModelComboAndScore(self, drug, analysis_files_folder, trimmed_cell_lines, trimmed_results, combos,
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
        best_model, trainer = self.trainBestModelWithCombo(best_scoring_algo, best_combo, optimal_hyperparams,
                                                           trimmed_cell_lines, trimmed_results, processed_arguments)
        return RecommendationsModelInfo(trainer, top_score, best_combo, best_model)

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
        params = DictionaryUtility.toDict(optimal_hyperparams)
        feature_names = training_matrix.get(ArgumentProcessingService.FEATURE_NAMES)
        model = trainer.buildModel(relevant_results, features, params, feature_names)
        return model, trainer

    def determineBestComboFromString(self, best_combo_string, combos, processed_arguments):
        gene_lists = processed_arguments.gene_lists
        combine_gene_lists = processed_arguments.rsen_config.combine_gene_lists
        analysis_type = processed_arguments.analysisType()
        for combo in combos:
            feature_set_string = GeneListComboUtility.generateFeatureSetString(combo, gene_lists,
                                                                               combine_gene_lists, analysis_type)
            if GeneListComboUtility.combosAreEquivalent(feature_set_string, best_combo_string):
                return combo

        raise ValueError("Unable to determine feature set from given combo gene list and feature file combo: " +
                         best_combo_string + ".\n Please make sure all gene lists and feature files in the combo " +
                         "are present in the drug folder.")

    def generateSinglePrediction(self, best_model, best_combo, cell_line, all_features, formatted_inputs):
        ommited_cell_line = formatted_inputs.get(DataFormattingService.TRAINING_MATRIX).get(cell_line)
        input_wrapper = OrderedDict()
        input_wrapper[DataFormattingService.TRAINING_MATRIX] = OrderedDict()
        input_wrapper[DataFormattingService.TRAINING_MATRIX][cell_line] = ommited_cell_line
        input_wrapper[ArgumentProcessingService.FEATURE_NAMES] = all_features
        trimmed_matrix = GeneListComboUtility.trimMatrixByFeatureSet(DataFormattingService.TRAINING_MATRIX,
                                                                     best_combo, input_wrapper,
                                                                     AnalysisType.RECOMMENDATIONS)
        return best_model.predict([trimmed_matrix.get(cell_line)])[0]
