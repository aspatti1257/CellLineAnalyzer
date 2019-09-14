import unittest
import os

from LoggerFactory import LoggerFactory
from Utilities.FileConverter import FileConverter


class FileConverterIT(unittest.TestCase):

    log = LoggerFactory.createLog(__name__)

    def setUp(self):
        current_working_dir = os.getcwd()  # Should be this package.
        self.input_folder = current_working_dir + "/SampleMatlabDataFolder"
        self.createdFolder = self.input_folder + "/Trametinib_analysis"

    def tearDown(self):
        if self.input_folder != "/":
            for file in os.listdir(self.createdFolder):
                if file == "__init__.py" or ".mat" in file:
                    continue
                os.remove(self.createdFolder + "/" + file)
            os.removedirs(self.createdFolder)

    def testMatlabFileConversionProperlyFormatsMatrices(self):
        FileConverter.convertMatLabToCSV(self.input_folder)
        for generated_csv in [file for file in os.listdir(self.createdFolder) if ".csv" in file]:
            with open(self.createdFolder + "/" + generated_csv) as csv:
                try:
                    for line in csv:
                        assert "['" not in line
                        assert "']" not in line
                except ValueError as valueError:
                    self.log.error(valueError)
                finally:
                    csv.close()
