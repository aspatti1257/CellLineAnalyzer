import scipy.io
import numpy as np
import logging

from Utilities.SafeCastUtil import SafeCastUtil


class FileConverter(object):

    @staticmethod
    def convertMatLabToCSV(input_file):
        mat_file = scipy.io.loadmat(input_file)

        log = logging.getLogger(__name__)
        logging.basicConfig()
        log.setLevel(logging.INFO)

        ctr = 0

        for key in mat_file:
            if '__' not in key and 'readme' not in key:
                save_name = 'convertedFile_' + SafeCastUtil.safeCast(key, str) + '.csv'
                np.savetxt(save_name, mat_file[key], fmt='%s', delimiter=',')
                log.info("Created CSV file %s", save_name)
                ctr += 1
