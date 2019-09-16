import numpy
from collections import OrderedDict
from itertools import repeat

from ArgumentConfig.AnalysisType import AnalysisType
from ArgumentProcessingService import ArgumentProcessingService
from Utilities.SafeCastUtil import SafeCastUtil


class GeneListComboUtility(object):

    ONLY_STATIC_FEATURES = "only static features"
    ALL_FEATURES = "all features"

    @staticmethod
    def determineCombos(gene_lists, feature_names, static_features):
        gene_sets_across_files = {}
        for feature in feature_names:
            split = feature.split(".")
            if split[0] in static_features:
                continue
            if gene_sets_across_files.get(split[0]) is not None:
                gene_sets_across_files[split[0]].append(feature)
            else:
                gene_sets_across_files[split[0]] = [feature]

        numerical_permutations = GeneListComboUtility.generateNumericalPermutations(gene_lists, gene_sets_across_files)
        gene_list_keys = SafeCastUtil.safeCast(gene_lists.keys(), list)
        file_keys = SafeCastUtil.safeCast(gene_sets_across_files.keys(), list)
        gene_list_combos = []
        for perm in numerical_permutations:
            feature_strings = []
            for static_feature_file in static_features:
                feature_strings.append([feature for feature in feature_names if static_feature_file in feature])
            for i in range(0, len(perm)):
                file_name = file_keys[i]
                gene_list = gene_lists[gene_list_keys[SafeCastUtil.safeCast(perm[i], int)]]
                if len(gene_list) > 0:
                    feature_strings.append([file_name + "." + gene for gene in gene_list if len(gene.strip()) > 0])
            if len(feature_strings) > 0:
                gene_list_combos.append(feature_strings)

        file_keys = SafeCastUtil.safeCast(gene_sets_across_files.keys(), list)
        gene_list_keys = SafeCastUtil.safeCast(gene_lists.keys(), list)
        expected_combo_length = (len(gene_list_keys) ** len(file_keys)) - 1
        if len(static_features) > 0:
            expected_combo_length += 1
        return gene_list_combos, expected_combo_length

    @staticmethod
    def generateNumericalPermutations(gene_lists, gene_sets_across_files):
        max_depth = len(gene_lists) - 1
        num_files = len(gene_sets_across_files)
        all_arrays = []
        current_array = SafeCastUtil.safeCast(numpy.zeros(num_files, dtype=numpy.int), list)
        target_index = num_files - 1
        while target_index >= 0:
            if current_array not in all_arrays:
                clone_array = current_array[:]
                all_arrays.append(clone_array)
            if current_array[target_index] < max_depth:
                current_array[target_index] += 1
                while len(current_array) > target_index + 1 and current_array[target_index + 1] < max_depth:
                    target_index += 1
            else:
                target_index -= 1
                for subsequent_index in range(target_index, len(current_array) - 1):
                    current_array[subsequent_index + 1] = 0
        return all_arrays

    @staticmethod
    def generateFeatureSetString(feature_set, gene_lists, combine_gene_lists, analysis_type, static_features):
        feature_map = {}
        for feature_list in feature_set:
            for feature in feature_list:
                file_name = feature.split(".")[0]
                feature_name = feature.split(".")[1:][0]
                if feature_map.get(file_name):
                    feature_map[file_name].append(feature_name)
                else:
                    feature_map[file_name] = [feature_name]

        feature_set_string = ""
        num_gene_lists_deduped = len(GeneListComboUtility.fetchAllGeneListGenesDeduped(gene_lists))
        for file_key in feature_map.keys():
            if combine_gene_lists and len(feature_map[file_key]) == num_gene_lists_deduped:
                feature_set_string += (file_key + ":ALL_GENE_LISTS ")
            else:
                for gene_list_key in gene_lists.keys():
                    if len(feature_map[file_key]) == len(gene_lists[gene_list_key]):
                        feature_map[file_key].sort()
                        gene_lists[gene_list_key].sort()
                        same_list = True
                        for i in range(0, len(gene_lists[gene_list_key])):
                            if gene_lists[gene_list_key][i] != feature_map[file_key][i]:
                                same_list = False
                        if same_list:
                            feature_set_string += (file_key + ":" + gene_list_key + " ")
        if feature_set_string == "" and analysis_type is AnalysisType.NO_GENE_LISTS:
            return GeneListComboUtility.ALL_FEATURES
        elif len(static_features) > 0 and sorted(feature_map.keys()) == sorted(static_features):
            return GeneListComboUtility.ONLY_STATIC_FEATURES
        return feature_set_string.strip()

    @staticmethod
    def fetchAllGeneListGenesDeduped(gene_lists):
        all_genes = SafeCastUtil.safeCast(gene_lists.values(), list)
        concated_genes = SafeCastUtil.safeCast(numpy.concatenate(all_genes), list)
        dedupded_genes = list(OrderedDict(zip(concated_genes, repeat(None))))
        return dedupded_genes

    @staticmethod
    def trimMatrixByFeatureSet(matrix_type, gene_lists, formatted_inputs, analysis_type):
        full_matrix = formatted_inputs.get(matrix_type)
        feature_names = formatted_inputs.get(ArgumentProcessingService.FEATURE_NAMES)

        if analysis_type is AnalysisType.NO_GENE_LISTS:
            # Skips an expensive process since we'll always be analyzing all features anyways in this mode.
            full_matrix[ArgumentProcessingService.FEATURE_NAMES] = feature_names
            return full_matrix

        trimmed_matrix = {
            ArgumentProcessingService.FEATURE_NAMES: []
        }

        important_indices = []
        feature_names = formatted_inputs.get(ArgumentProcessingService.FEATURE_NAMES)
        for i in range(0, len(feature_names)):
            for gene_list in gene_lists:
                for gene in gene_list:
                    if gene == feature_names[i]:
                        important_indices.append(i)
                        trimmed_matrix[ArgumentProcessingService.FEATURE_NAMES].append(gene)

        for cell_line in full_matrix.keys():
            new_cell_line_features = []
            for j in range(0, len(full_matrix[cell_line])):
                if j in important_indices:
                    new_cell_line_features.append(full_matrix[cell_line][j])
            trimmed_matrix[cell_line] = new_cell_line_features
        return trimmed_matrix


    @staticmethod
    def combosAreEquivalent(combo_one, combo_two):
        return sorted(combo_one.split(" ")) == sorted(combo_two.split(" "))
