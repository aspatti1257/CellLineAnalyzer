from collections import OrderedDict
from Utilities.SafeCastUtil import SafeCastUtil


class DictionaryUtility(object):

    @staticmethod
    def toDict(dict_as_string):
        dictionary = OrderedDict()
        split_dict = dict_as_string.split(",")
        for key_val_pair in split_dict:
            as_tuple = SafeCastUtil.safeCast(key_val_pair.split(":"), tuple)
            dictionary[as_tuple[0].strip()] = SafeCastUtil.safeCast(as_tuple[1].strip(), float, as_tuple[1].strip())
        return dictionary

    @staticmethod
    def toString(dictionary):
        hyperparam_string = ""
        keys = SafeCastUtil.safeCast(dictionary.keys(), list)
        for i in range(0, len(keys)):
            hyperparam_string += (keys[i] + ": " + SafeCastUtil.safeCast(dictionary[keys[i]], str))
            if i < len(keys) - 1:
                hyperparam_string += ", "
        return hyperparam_string
