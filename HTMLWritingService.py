import logging
import os

from Utilities.SafeCastUtil import SafeCastUtil
from Trainers.AbstractModelTrainer import AbstractModelTrainer


class HTMLWritingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    RECORD_FILE = "FullResultsSummary.txt"
    SUMMARY_FILE = "SummaryReport.html"

    def __init__(self, input_folder):
        self.input_folder = input_folder

    def writeSummaryFile(self):
        self.createStatsOverviewFromFile()

    def createStatsOverviewFromFile(self):
        stats_overview_object = self.generateStatsOverviewObject()
        new_file = self.generateNewReportFile(stats_overview_object)

        with open(self.input_folder + "/" + self.SUMMARY_FILE, "w") as summary_file:
            try:
                for line in new_file:
                    summary_file.write(line)
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                summary_file.close()

        self.log.info(new_file)

    def generateStatsOverviewObject(self):
        stats_overview_object = {}
        with open(self.input_folder + "/" + self.RECORD_FILE) as record_file:
            try:
                for line_index, line in enumerate(record_file):
                    line_split = [segment.strip() for segment in line.split("---")]
                    self.log.info(line_split)
                    if len(line_split) < 3:
                        self.log.warning("Line from results file not split properly: %s", line)
                        continue

                    scores = self.translateToNumericList(line_split[2])
                    accuracies = self.translateToNumericList(line_split[3])
                    if stats_overview_object.get(line_split[0]) is None:
                        stats_overview_object[line_split[0]] = {line_split[1]: [scores, accuracies]}
                    else:
                        stats_overview_object[line_split[0]][line_split[1]] = [scores, accuracies]
            except ValueError as value_error:
                self.log.error(value_error)
            finally:
                record_file.close()
        return stats_overview_object

    def translateToNumericList(self, line_split):
        return [SafeCastUtil.safeCast(val, float) for val in line_split.replace("[", "").replace("]", "").split(",")]

    def generateNewReportFile(self, stats_overview_object):
        path_of_this_file = os.path.realpath(__file__)
        template_path = os.path.abspath(os.path.join(path_of_this_file, os.pardir)) + "/Reports/reportTemplate.html"
        new_file = []
        with open(template_path) as template_file:
            try:
                for line_index, line in enumerate(template_file):
                    if "//INSERT DEFAULT MIN SCORE HERE" in line:
                        new_file.append("\t\t\t\tvar DEFAULT_MIN_SCORE = " +
                                        SafeCastUtil.safeCast(AbstractModelTrainer.DEFAULT_MIN_SCORE, str) + ";\n")
                    elif "//INSERT CHART DATA HERE" in line:
                        new_file.append("\t\t\t\t$scope.allData = " +
                                        SafeCastUtil.safeCast(stats_overview_object, str) + ";\n")
                    else:
                        new_file.append(line)
            except ValueError as valueError:
                self.log.error(valueError)
            finally:
                template_file.close()
        return new_file
