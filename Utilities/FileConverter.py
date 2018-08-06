import scipy.io
import logging
import csv
import glob
import os

from Utilities.SafeCastUtil import SafeCastUtil


class FileConverter(object):

    VARIABLE_MATCHES = {
        "genesCNHugo": "gmcCN",
        "genesExpHugo": "gmcGE",
        "genesMutHugo": "gmcMUT"
    }

    FILE_NAMES = {
        "genesCNHugo": "cnum_hgnc",
        "genesExpHugo": "gex_hgnc",
        "genesMutHugo": "mut_hgnc"
    }

    EXPECTED_TYPES = {
        "genesCNHugo": int,
        "genesExpHugo": float,
        "genesMutHugo": str
    }


    ID_FIELD = "gmcCellLineCosmicIDs"
    RESULTS_FIELD = "gmcAUC"

    @staticmethod
    def convertMatLabToCSV(matlab_files_directory):

        log = logging.getLogger(__name__)
        logging.basicConfig()
        log.setLevel(logging.INFO)

        os.chdir(matlab_files_directory)
        matlab_files = glob.glob("*.mat")

        for input_file in matlab_files:
            drug_name = input_file.split("gexmutcnum.mat")[0].strip()
            new_directory = matlab_files_directory + "/" + drug_name + "_analysis"
            matlab_file = scipy.io.loadmat(input_file)

            os.mkdir(new_directory)

            format_id_string = lambda array: SafeCastUtil.safeCast(array[0], str)
            for key in SafeCastUtil.safeCast(FileConverter.VARIABLE_MATCHES.keys(), list):
                header = [format_id_string(feature_name) for feature_name in matlab_file.get(key)[0]]
                file_name = new_directory + "/" + drug_name + "_" + FileConverter.FILE_NAMES[key] + ".csv"
                cell_line_data = FileConverter.formatCellLineData(
                                    matlab_file.get(FileConverter.VARIABLE_MATCHES.get(key)), key)
                FileConverter.validateAndWriteCSV(cell_line_data, header, file_name, log, FileConverter.EXPECTED_TYPES[key])

            cell_line_ids = [format_id_string(cell_id) for cell_id in matlab_file.get(FileConverter.ID_FIELD)]
            results = matlab_file.get(FileConverter.RESULTS_FIELD)
            zipped_results = SafeCastUtil.safeCast(zip(cell_line_ids, results[0]), list)
            results_file = new_directory + "/" + drug_name + "_results.csv"

            FileConverter.validateAndWriteCSV(zipped_results, ["cell_line", "result"], results_file, log, float)
            log.info("The MATLAB file for %s has been successfully converted into csv files ready to be used"
                     " with the CLA software!", drug_name)

        log.info("All MATLAB files have been processed!")

    @staticmethod
    def formatCellLineData(data, key):
        if "CN" in key:
            return FileConverter.reduceCopyNumFile(data)
        elif "Mut" in key:
            return [["'" + SafeCastUtil.safeCast(value, str) + "'" for value in row] for row in data]
        else:
            return data

    @staticmethod
    def reduceCopyNumFile(cell_line_data):
        for row in cell_line_data:
            for index, element in enumerate(row):
                element_as_string = SafeCastUtil.safeCast(element, str)
                if (len(element_as_string) == 1):
                    row[index] = 0
                elif (len(element_as_string) == 2):
                    row[index] = 2
                elif (len(element_as_string) == 3):
                    row[index] = element_as_string[0]
                elif (len(element_as_string) == 4):
                    row[index] = element_as_string[0:2]
                elif (len(element_as_string) == 5):
                    row[index] = element_as_string[1]
                elif (len(element_as_string) == 6):
                    row[index] = element_as_string[2]
                elif (len(element_as_string) == 7):
                    row[index] = element_as_string[2:4]
        return cell_line_data

    @staticmethod
    def validateAndWriteCSV(data, first_line, file_name, log, expected_type):
        for row in data:
            for element in row:
                try:
                    expected_type(element)
                except TypeError as error:
                    log.error("Unable to cast %s to %s.", element, expected_type)
                    return

        with open(file_name, 'w', newline='') as csv_file:
            try:
                writer = csv.writer(csv_file)
                writer.writerow(first_line)
                for cell_line in data:
                    writer.writerow(SafeCastUtil.safeCast(cell_line, list))
            except ValueError as error:
                log.error("Error writing to file %s. %s", file_name, error)
            finally:
                csv_file.close()