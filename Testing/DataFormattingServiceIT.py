import unittest
import numpy as np
import os

class DataFormattingServiceIT(unittest.TestCase):
    #  TODO - Christine: Tests for DataFormattingService. You should be able load the class directly and build out
    #  methods to test the classes directly.

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def checkimportdata(self):
        features = np.genfromtxt('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = np.genfromtxt('SampleClassifierDataFolder/results.csv', delimiter=',')
        assert np.array(features[1:]).dtype == "float64"
        assert np.array(results[1:,1]).dtype == "float64"
        assert np.isnan(features[1:]).any() == False
        assert np.isnan(results[1:,1]).any() == False
        assert len(features) == len(results)


d = DataFormattingServiceIT()
print(d.checkimportdata())