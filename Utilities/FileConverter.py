import scipy.io
import numpy
import logging
import os

from Utilities.SafeCastUtil import SafeCastUtil


class FileConverter(object):

    @staticmethod
    def convertMatLabToCSV(input_file):
        log = logging.getLogger(__name__)
        logging.basicConfig()
        log.setLevel(logging.INFO)

        counter = 0

        path = os.path.abspath(os.path.join(input_file, os.pardir))

        mat_file = scipy.io.loadmat(input_file)

        for key in mat_file:
            if '__' not in key and 'readme' not in key:
                save_name = path + "/convertedFile_" + SafeCastUtil.safeCast(key, str) + ".csv"
                original_matrix = mat_file[key]
                numpy.savetxt(save_name, FileConverter.cleanMatrix(original_matrix), fmt='%s', delimiter=',')
                log.info("Created CSV file %s", save_name)
                counter += 1

        log.info("The csv files have been created!")

    @staticmethod
    def cleanMatrix(original_matrix):
        cleaned_matrix = []
        for row in [row for row in original_matrix if type(row) == numpy.ndarray]:
            cleaned_row = []
            for element in [element for element in row if type(element) == numpy.ndarray and len(element) == 1]:
                cleaned_row.append(element[0])
            cleaned_matrix.append(cleaned_row)
        return cleaned_matrix
