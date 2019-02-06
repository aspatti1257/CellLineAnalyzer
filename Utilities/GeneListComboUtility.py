from Utilities.SafeCastUtil import SafeCastUtil
import numpy


class GeneListComboUtility(object):

    @staticmethod
    def determineGeneListCombos(gene_lists, feature_names):
        gene_sets_across_files = {}
        for feature in feature_names:
            split = feature.split(".")
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
