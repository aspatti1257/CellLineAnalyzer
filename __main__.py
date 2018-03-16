import sys
import logging

from ArgumentProcessingService import ArgumentProcessingService
from DataFormattingService import DataFormattingService
from MachineLearningService import MachineLearningService
from Utilities.SafeCastUtil import SafeCastUtil

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)


def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        promptUserForInput()
    elif len(arguments) == 1:
        input_folder = arguments[0]
        runMainCellLineAnalysis(input_folder)
    return


def promptUserForInput():
    simulation_to_run = input("-------Main Menu-------\n"
                              "Choose your task:\n"
                              "\t0: Analysis of cell lines\n"
                              "\tQ: Quit\n")

    simulation_as_int = SafeCastUtil.safeCast(simulation_to_run, int)
    simulation_as_string = SafeCastUtil.safeCast(simulation_to_run, str, "Q")

    if simulation_as_string == "Q":
        return
    elif simulation_as_int == 0:
        input_folder = recursivelyPromptUser("Enter path of input folder:\n", str)
        runMainCellLineAnalysis(input_folder)


def runMainCellLineAnalysis(input_folder):
    valid_inputs = handleInputFolderProcessing(input_folder)
    if valid_inputs is not None:
        formatted_data = handleDataFormatting(valid_inputs)
        performMachineLearning(formatted_data, input_folder)


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


def handleDataFormatting(inputs):
    data_formatting_service = DataFormattingService(inputs)
    return data_formatting_service.formatData()


def performMachineLearning(formatted_data, input_folder):
    machine_learning_service = MachineLearningService(formatted_data)
    machine_learning_service.analyze(input_folder)


if __name__ == "__main__":
    main()
