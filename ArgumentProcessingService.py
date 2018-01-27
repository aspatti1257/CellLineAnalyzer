import os
import logging
import re

from Utilities.SafeCastUtil import SafeCastUtil


class ArgumentProcessingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    ARGUMENTS_FILE = "arguments.txt"

    def __init__(self, input_folder):
        self.input_folder = input_folder
        pass

    def handleInputFolder(self):
        directory_contents = os.listdir(self.input_folder)
        if not self.validateDirectoryContents(directory_contents):
            self.log.error("Invalid directory contents, needs a %s file.",
                           self.ARGUMENTS_FILE)
            return None

        arguments = self.fetchArguments(self.input_folder + "/" + self.ARGUMENTS_FILE)
        results_file = arguments.get("results")
        is_classifier = SafeCastUtil.safeCast(arguments.get("is_classifier"), int) == 1
        if is_classifier is not None and results_file is not None:
            results_list = self.validateAndExtractResults(results_file, is_classifier)
            self.validateFeatureFiles(results_list, results_file)  # TODO: Return feature matrix and important features.
            return {
                "results": results_list,
                "is_classifier": is_classifier
            }
        else:
            return None

    def validateDirectoryContents(self, directory_contents):
        return self.ARGUMENTS_FILE in directory_contents

    def fetchArguments(self, arguments_file):
        arguments = {}
        with open(arguments_file) as data_file:
            try:
                for line in data_file:
                    line_trimmed_split = line.replace("\n", "").split("=")
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
                    line_trimmed_split = line.replace("\n", "").split(",")
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

    def validateFeatureFiles(self, results_list, results_file):
        files = os.listdir(self.input_folder)
        num_results = len(results_list)
        for file in [file for file in files if file != results_file and file != self.ARGUMENTS_FILE]:
            if self.findLineCountSimple(file) != (num_results + 1):
                self.log.error("Invalid line count for %s", num_results)
                raise ValueError("Invalid line count for" + file + ". Must be " +
                                 SafeCastUtil.safeCast(num_results, str) + "lines long.")

    def findLineCountSimple(self, file):  # TODO: Maybe auto-trim empty lines?
        file_path = self.input_folder + "/" + file
        return sum(1 for line in open(file_path))
