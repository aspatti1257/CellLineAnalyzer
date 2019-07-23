import numpy


class PercentageBarUtility:

    @staticmethod
    def calculateAndCreatePercentageBar(numerator, denominator):
        percent_done = numpy.round((numerator / denominator) * 100, 1)
        percentage_bar = "["
        for i in range(0, 100):
            if i < percent_done:
                percentage_bar += "="
            elif i >= percent_done:
                if i > 0 and (i - 1) < percent_done:
                    percentage_bar += ">"
                else:
                    percentage_bar += " "
        percentage_bar += "]"
        return percent_done, percentage_bar
