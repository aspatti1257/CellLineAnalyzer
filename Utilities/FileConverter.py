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
        "genesMutHugo": "gmcMUT",
        "genesCNEnsembl": "gmcCN", #TODO: These Ensemble features seem like they should be removed.
        "genesExpEnsembl": "gmcGE",
        "genesMutEnsembl": "gmcMUT"
    }

    FILE_NAMES = {
        "genesCNHugo": "cnum_hgnc",
        "genesExpHugo": "gex_hgnc",
        "genesMutHugo": "mut_hgnc",
        "genesCNEnsembl": "cnum_ensg", #TODO: These Ensemble features seem like they should be removed.
        "genesExpEnsembl": "gex_ensg",
        "genesMutEnsembl": "mut_ensg"
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
                cell_line_data = FileConverter.formatCellLineData(
                                    matlab_file.get(FileConverter.VARIABLE_MATCHES.get(key)), key)
                file_name = new_directory + "/" + drug_name + "_" + FileConverter.FILE_NAMES[key] + ".csv"
                FileConverter.writeCSV(cell_line_data, header, file_name, log)

            cell_line_ids = [format_id_string(cell_id) for cell_id in matlab_file.get(FileConverter.ID_FIELD)]
            results = matlab_file.get(FileConverter.RESULTS_FIELD)
            zipped_results = SafeCastUtil.safeCast(zip(cell_line_ids, results[0]), list)
            results_file = new_directory + "/" + drug_name + "_results.csv"

            FileConverter.writeCSV(zipped_results, ["cell_line", "result"], results_file, log)
            log.info("The MATLAB file for %s has been successfully converted into csv files ready to be used"
                     " with the CLA software!", drug_name)

        log.info("All MATLAB files have been processed!")

    @staticmethod
    def formatCellLineData(data, key):
        if "Mut" not in key and "CN" not in key:
            return data
        return [["'" + SafeCastUtil.safeCast(value, str) + "'" for value in row] for row in data]

    @staticmethod
    def writeCSV(data, first_line, file_name, log):
        with open(file_name, "w") as csv_file:
            try:
                writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(first_line)
                for cell_line in data:
                    writer.writerow(SafeCastUtil.safeCast(cell_line, list))
            except ValueError as error:
                log.error("Error writing to file %s. %s", file_name, error)
            finally:
                csv_file.close()
