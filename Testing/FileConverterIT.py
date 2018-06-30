import unittest
import logging
import os

from Utilities.FileConverter import FileConverter


class FileConverterIT(unittest.TestCase):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def setUp(self):
        current_working_dir = os.getcwd()  # Should be this package.
        self.input_folder = current_working_dir + "/SampleMatlabDataFolder"

    def tearDown(self):
        if self.input_folder != "/":
            for file in os.listdir(self.input_folder):
                if file == "__init__.py" or ".mat" in file:
                    continue
                os.remove(self.input_folder + "/" + file)

    def testMatlabFileConversionProperlyFormatsMatrices(self):
        FileConverter.convertMatLabToCSV(self.input_folder + "/gexmutSepantroniumbromide.mat")
        for generated_csv in [file for file in os.listdir(self.input_folder) if "convertedFile_" in file]:
            with open(self.input_folder + "/"  + generated_csv) as csv:
                try:
                    for line in csv:
                        assert "['" not in line
                        assert "']" not in line
                except ValueError as valueError:
                    self.log.error(valueError)
                finally:
                    csv.close()
