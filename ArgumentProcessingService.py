import os
import logging
import re

from Utilities.SafeCastUtil import SafeCastUtil


class ArgumentProcessingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    ARGUMENTS_FILE = "arguments.txt"
    GENE_LISTS = "gene_list"

    RESULTS = "results"
    IS_CLASSIFIER = "is_classifier"
    FEATURES = "features"
    FEATURE_NAMES = "featureNames"
    IMPORTANT_FEATURES = "important_features"  # TODO: Maybe regress

    def __init__(self, input_folder):
        self.input_folder = input_folder

    def handleInputFolder(self):
        directory_contents = os.listdir(self.input_folder)
        if not self.validateDirectoryContents(directory_contents):
            self.log.error("Invalid directory contents, needs a %s file.",
                           self.ARGUMENTS_FILE)
            return None

        arguments = self.fetchArguments(self.input_folder + "/" + self.ARGUMENTS_FILE)
        results_file = arguments.get(self.RESULTS)
        is_classifier = SafeCastUtil.safeCast(arguments.get(self.IS_CLASSIFIER), int) == 1
        important_features = self.extractImportantFeatures(arguments)
        if is_classifier is not None and results_file is not None:
            results_list = self.validateAndExtractResults(results_file, is_classifier)
            feature_map = self.createAndValidateFeatureMatrix(results_list, results_file, important_features)
            gene_lists = self.extractGeneList()
            return {
                self.RESULTS: results_list,
                self.IS_CLASSIFIER: is_classifier,
                self.FEATURES: feature_map,
                self.GENE_LISTS: gene_lists
            }
        else:
            return None

    def extractImportantFeatures(self, arguments):
        important_features = arguments.get(self.IMPORTANT_FEATURES)
        if important_features is not None:
            return [feature.strip() for feature in important_features.split(",")]
        else:
            return []

    def validateDirectoryContents(self, directory_contents):
        return self.ARGUMENTS_FILE in directory_contents

    def fetchArguments(self, arguments_file):
        arguments = {}
        with open(arguments_file) as data_file:
            try:
                for line in data_file:
                    line_trimmed_split = line.strip().split("=")
                    arguments[line_trimmed_split[0]] = line_trimmed_split[1]
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                self.log.debug("Closing file %s", arguments_file)
                data_file.close()
        return arguments

    def validateAndExtractResults(self, results_file, is_classifier):
        sample_list = []
        cast_type = float
        if is_classifier:
            cast_type = int
        results_path = self.input_folder + "/" + results_file
        with open(results_path) as data_file:
            try:
                for line_index, line in enumerate(data_file):
                    if len(re.findall(r'^\s*$', line)) > 0 or line_index == 0:  # header or whitespace
                        continue
                    line_trimmed_split = line.strip().split(",")
                    if len(line_trimmed_split) != 2:
                        self.log.error("Each line in %s must be 2 columns. Aborting argument processing.",
                                       results_file)
                        raise ValueError("Each line in results file must be 2 columns.")
                    cell_line = line_trimmed_split[0]
                    cell_result = SafeCastUtil.safeCast(line_trimmed_split[1], cast_type)
                    if cell_line in sample_list:
                        self.log.error("Repeated cell line name: %s. Aborting argument processing.", cell_line)
                        raise ValueError("Repeated cell line name.")
                    else:
                        sample_list.append([cell_line, cell_result])
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                self.log.debug("Closing file %s", results_file)
                data_file.close()
        return sample_list

    def createAndValidateFeatureMatrix(self, results_list, results_file, important_features):
        files = os.listdir(self.input_folder)
        feature_map = {self.FEATURE_NAMES: []}
        for file in [f for f in files if f != results_file and f != self.ARGUMENTS_FILE and self.GENE_LISTS not in f]:
            features_path = self.input_folder + "/" + file
            with open(features_path) as feature_file:
                try:
                    important_feature_indices = []
                    for line_index, line in enumerate(feature_file):
                        if line_index == 0:
                            feature_names = line.split(",")
                            for i in range(0, len(feature_names)):
                                feature_name = self.determineFeatureName(feature_names[i], file)
                                if self.featureIsImportant(important_features, feature_name):
                                    feature_map[self.FEATURE_NAMES].append(feature_name)
                                    important_feature_indices.append(i)
                        else:
                            features = self.extractCastedFeatures(line, important_feature_indices)
                            cell_line = results_list[line_index - 1]
                            if not cell_line[0] in feature_map:
                                feature_map[cell_line[0]] = features
                            else:
                                feature_map[cell_line[0]] = feature_map[cell_line[0]] + features
                            if line_index > len(results_list):
                                self.log.error("Invalid line count for %s", file)
                                raise ValueError("Invalid line count for" + file + ". Must be " +
                                                 SafeCastUtil.safeCast(file, str) + "lines long.")
                except ValueError as valueError:
                    self.log.error(valueError)
                    return None
                finally:
                    self.log.debug("Closing file %s", feature_file)
        return feature_map

    def determineFeatureName(self, feature_name, file):
        return SafeCastUtil.safeCast(file.split(".")[0] + "." + feature_name.strip(), str)

    def featureIsImportant(self, important_features, feature_name):
        if len(important_features) == 0:
            return True
        for feature in important_features:
            if feature == feature_name:
                return True
        return False

    def extractCastedFeatures(self, line, important_feature_indices):
        important_features = []
        feature_names = line.split(",")
        for i in range(0, len(feature_names)):
            if i in important_feature_indices:
                if SafeCastUtil.safeCast(feature_names[i], float) is not None:
                    important_features.append(SafeCastUtil.safeCast(feature_names[i].strip(), float))
                else:
                    important_features.append(SafeCastUtil.safeCast(feature_names[i].strip(), str))
        return important_features

    def extractGeneList(self):
        gene_lists = {"null_gene_list": []}
        files = os.listdir(self.input_folder)
        for file in [f for f in files if self.GENE_LISTS in f]:
            file_path = self.input_folder + "/" + file
            with open(file_path) as gene_list_file:
                gene_lists[file.split(".csv")[0]] = gene_list_file.read().strip().split(",")

        return gene_lists
