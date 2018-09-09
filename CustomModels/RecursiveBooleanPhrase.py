from Utilities.SafeCastUtil import SafeCastUtil


class RecursiveBooleanPhrase:

    def __init__(self, split, value, is_or, nested_phrase):
        self.split = split
        self.value = value
        self.is_or = is_or
        self.nested_phrase = nested_phrase

    def analyzeForFeatureSet(self, feature_set):
        if self.split is None or self.value is None:
            return True  # Fallback phrase, match all

        current_statement = (feature_set[self.split] == self.value)

        if self.nested_phrase is not None:
            nested_value = self.nested_phrase.analyzeForFeatureSet(feature_set)
            if self.is_or:
                return current_statement or nested_value
            else:
                return current_statement and nested_value
        return current_statement

    def toSummaryString(self):
        summary = "(Feature " + (SafeCastUtil.safeCast(self.split, str) + " == " + SafeCastUtil.safeCast(self.value, str))
        if self.nested_phrase is not None:
            if self.is_or:
                summary += " OR "
            else:
                summary += " AND "
            summary += self.nested_phrase.asSummaryString()

        return summary + ")"
