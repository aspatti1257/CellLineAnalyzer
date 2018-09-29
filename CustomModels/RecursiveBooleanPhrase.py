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
        summary = "(Feature " + (str(self.split) + " == " + str(self.value))
        if self.nested_phrase is not None:
            if self.is_or:
                summary += " OR "
            else:
                summary += " AND "
            summary += self.nested_phrase.toSummaryString()

        return summary + ")"

    def equals(self, other_phrase):
        attributes_are_equal = self.split == other_phrase.split and self.value == other_phrase.value and\
                               self.is_or == other_phrase.is_or
        if not attributes_are_equal or (self.nested_phrase is None and other_phrase.nested_phrase is not None) or\
                (self.nested_phrase is not None and other_phrase.nested_phrase is None):
            return False
        if self.nested_phrase is not None and self.nested_phrase is not None:
            return self.nested_phrase.equals(other_phrase.nested_phrase)
        return True

    def isValid(self, binary_feature_indices):
        if self.split is None and self.nested_phrase is None:
            return True
        is_valid = self.split in binary_feature_indices
        if self.nested_phrase is not None:
            return is_valid and self.nested_phrase.isValid(binary_feature_indices)
        return is_valid
