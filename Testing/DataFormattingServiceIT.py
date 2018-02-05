import unittest
import numpy as np
import os


class DataFormattingServiceIT(unittest.TestCase):

    def setUp(self):
        self.current_working_dir = os.getcwd()  # Should be this package.

    def tearDown(self):
        pass

    def testCheckImportData(self):
        features = np.genfromtxt('SampleClassifierDataFolder/features.csv', delimiter=',')
        results = np.genfromtxt('SampleClassifierDataFolder/results.csv', delimiter=',')
        assert np.array(features[1:]).dtype == "float64"
        assert np.array(results[1:, 1]).dtype == "float64"
        assert not np.isnan(features[1:]).any()
        assert not np.isnan(results[1:, 1]).any()
        assert len(features) == len(results)

