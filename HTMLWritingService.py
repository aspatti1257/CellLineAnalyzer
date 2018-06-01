import logging

from Utilities.SafeCastUtil import SafeCastUtil


class HTMLWritingService(object):

    log = logging.getLogger(__name__)
    logging.basicConfig()
    log.setLevel(logging.INFO)

    RECORD_FILE = "FullResultsSummary.txt"

    def __init__(self, input_folder):
        self.input_folder = input_folder

    def writeSummaryFile(self):
        stats_overview_object = self.createStatsOverviewFromFile()
        self.log.debug(stats_overview_object)
        #TODO: Write out AngularJS template file with d3 charts.

    def createStatsOverviewFromFile(self):
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

