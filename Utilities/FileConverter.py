import scipy.io
import numpy as np
import logging
import csv
import pandas as pd
import glob
import os

from Utilities.SafeCastUtil import SafeCastUtil


class FileConverter(object):

    @staticmethod
    def convertMatLabToCSV(matlab_files_directory):
        os.chdir(matlab_files_directory)
        log = logging.getLogger(__name__)
        logging.basicConfig()
        log.setLevel(logging.INFO)
        matlab_files = glob.glob("*.mat")

        for input_file in matlab_files:
            matlab_file = scipy.io.loadmat(input_file)
            ctr = 0
            for key in matlab_file:
                if "__" not in key and "readme" not in key:
                    savename = input_file.replace(".mat", "") + "_" + str(ctr) + ".csv"
                    np.savetxt(savename, matlab_file[key], fmt='%s', delimiter=',')
                    with open(savename, "r") as source:
                        data = source.read()
                        replace = {"['": "", "']": ""}
                        for x, y in replace.items():
                            data = data.replace(x, y)
                        with open(savename, "w") as savename_write:
                            savename_write.write(data)
                            savename_write.close()
                    ctr += 1

            # second step: transpose _6 csv file and remove first row (empty row after transpose)

            read_csv = pd.read_csv(input_file.replace(".mat", "") + "_6.csv", nrows=1)
            transpose = np.transpose(read_csv)

            df = pd.DataFrame(transpose)
            df.to_csv(input_file.replace(".mat", "") + "_6.csv")

            with open(input_file.replace(".mat", "") + "_6.csv", "rb") as infile:
                data_in = infile.readlines()
                with open(input_file.replace(".mat", "") + "_6.csv", "wb") as outfile:
                    outfile.writelines(data_in[1:])

            # third step: merge the transposed file with the list of cell lines (with headers)

            csv_8 = pd.read_csv(input_file.replace(".mat", "") + "_8.csv")
            csv_6_transpose = pd.read_csv(input_file.replace(".mat", "") + "_6.csv")
            merge = pd.concat([csv_8, csv_6_transpose], axis=1)
            merge.to_csv(input_file.replace("gexmutcnum.mat", "") + "_results.csv", index=False)

            with open(input_file.replace("gexmutcnum.mat", "") + "_results.csv", "r+") as header_to_results:
                old = header_to_results.read()
                header_to_results.seek(0)  # rewind
                header_to_results.write("cell_line, result" + "\n" + old)

            # fourth step: convert categorical data in _7 and _10 csv file into strings

            def categoriser(a):
                with open(input_file.replace(".mat", "") + a, "r") as f:
                    reader_f = csv.reader(f)
                    liste = list(reader_f)

                new_list = []
                for row in liste:
                    new_list.append([str("'" + val + "'") for val in row])

                with open(input_file.replace(".mat", "") + a, "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(new_list)

            categoriser("_7.csv")
            categoriser("_10.csv")

            # fifth step: merge all the other files

            def merger(a, b, c):
                with open(input_file.replace(".mat", "") + a, "r") as a1:
                    reader_a = csv.reader(a1)
                    a_list = list(reader_a)

                with open(input_file.replace(".mat", "") + b, "r") as b1:
                    reader_b = csv.reader(b1)
                    b_list = list(reader_b)

                merged = a_list + b_list

                with open(input_file.replace("gexmutcnum.mat", "") + c, "w") as merged_output:
                    writer = csv.writer(merged_output)
                    writer.writerows(merged)

            merger("_0.csv", "_7.csv", "_cnum_ensg.csv")
            merger("_1.csv", "_7.csv", "_cnum_hgnc.csv")
            merger("_2.csv", "_9.csv", "_gex_ensg.csv")
            merger("_3.csv", "_9.csv", "_gex_hgnc.csv")
            merger("_4.csv", "_10.csv", "_mut_ensg.csv")
            merger("_5.csv", "_10.csv", "_mut_hgnc.csv")

            # sixth step: create new directory for the analysis data

            new_folder = os.mkdir(input_file.replace("gexmutcnum.mat", "") + "_analysis")
            new_folder

            def directoriser(a):
                path = input_file.replace("gexmutcnum.mat", "") + "_analysis/" + input_file.replace("gexmutcnum.mat", "") + a
                new_directory = ("%s" % path)
                os.rename(input_file.replace("gexmutcnum.mat", "") + a, new_directory)

            directoriser("_cnum_ensg.csv")
            directoriser("_cnum_hgnc.csv")
            directoriser("_gex_ensg.csv")
            directoriser("_gex_hgnc.csv")
            directoriser("_mut_ensg.csv")
            directoriser("_mut_hgnc.csv")
            directoriser("_results.csv")

            # seventh step: delete the original csv files that have been converted from the inital matlab file

            def csv_eliminator(a):
                os.remove(input_file.replace(".mat", "") + a)

            for i in range(0, 11):
                csv_eliminator("_%s.csv" % i)

            # eighth step: finished

            log.info("The matlab file for %s has been successfully converted into csv files ready to be used for the CLA software!" % input_file.replace("gexmutcnum.mat", ""))

    log = logging.getLogger(__name__)
    log.info("All matlab files have been processed!")

