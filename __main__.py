import sys
import logging
import os

from ArgumentProcessingService import ArgumentProcessingService
from MachineLearningService import MachineLearningService
from HTMLWritingService import HTMLWritingService
from RecommendationsService import RecommendationsService
from Utilities.SafeCastUtil import SafeCastUtil
from Utilities.FileConverter import FileConverter

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)


def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        promptUserForInput()
    elif len(arguments) == 2 and arguments[0] == '0':
        runMainCellLineAnalysis(arguments[1])
    elif len(arguments) == 2 and arguments[0] == '1':
        FileConverter.convertMatLabToCSV(arguments[1])
    elif len(arguments) == 2 and arguments[0] == '2':
        fetchRecommendations(arguments[1])
    else:
        log.error("Exiting program, invalid data sent in target folder.")
    return


def promptUserForInput():
    simulation_to_run = input("-------Main Menu-------\n"
                              "Choose your task:\n"
                              "\t0: Analysis of cell lines\n"
                              "\t1: Convert MATLAB to CSV file\n"
                              "\t2: Dr.S Analysis (Drug Recommendations System)"
                              "\tQ: Quit\n")

    option_as_int = SafeCastUtil.safeCast(simulation_to_run, int)
    option_as_string = SafeCastUtil.safeCast(simulation_to_run, str, "Q")

    if option_as_string == "Q":
        return
    elif option_as_int == 0:
        input_folder = recursivelyPromptUser("Enter path of input folder:\n", str)
        runMainCellLineAnalysis(input_folder)
    elif option_as_int == 1:
        matlab_files_directory = recursivelyPromptUser("Enter folder path of the matlab files:\n", str)
        FileConverter.convertMatLabToCSV(matlab_files_directory)
    elif option_as_int == 2:
        input_folder = recursivelyPromptUser("Enter folder path of the input folder:\n", str)
        fetchRecommendations(input_folder)


def runMainCellLineAnalysis(input_folder):
    valid_inputs = handleInputFolderProcessing(input_folder)
    if valid_inputs is not None:
        performMachineLearning(valid_inputs, input_folder)
        is_classifier = valid_inputs.is_classifier
        writeHTMLSummaryFile(input_folder, is_classifier)


def recursivelyPromptUser(message, return_type):
    response = input(message)
    cast_response = SafeCastUtil.safeCast(response, return_type)
    if cast_response is None:
        print("Invalid command, looking for an input of type %.\n", return_type)
        recursivelyPromptUser(message, return_type)
    else:
        return response


def handleInputFolderProcessing(input_folder):
    argument_processing_service = ArgumentProcessingService(input_folder)
    full_inputs = argument_processing_service.handleInputFolder()
    if not full_inputs:
        log.error("Exiting program, invalid data sent in target folder.")
        return None
    return full_inputs


def performMachineLearning(valid_inputs, input_folder):
    machine_learning_service = MachineLearningService(valid_inputs)
    return machine_learning_service.analyze(input_folder)


def writeHTMLSummaryFile(input_folder, is_classifier):
    html_writing_service = HTMLWritingService(input_folder, is_classifier)
    html_writing_service.writeSummaryFile()


def fetchRecommendations(input_folder):
    processed_args = {}
    for drug_dir in [file for file in os.listdir(input_folder) if os.path.isdir(input_folder + file)]:
        processed_args[drug_dir] = handleInputFolderProcessing(input_folder)
    recs_service = RecommendationsService(processed_args)
    recs_service.recommendByHoldout(input_folder)


if __name__ == "__main__":
    main()
