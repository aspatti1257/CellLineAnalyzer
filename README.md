# CellLineAnalyzer - Introduction

The Cell Line analyzer is designed to create machine learning models on large scale bioinformatics data, and
determine which combination of features are the best predictors for a particular outcome. The inputs to this program
are:

- A series of .CSV files indicating important features or genes, broken up and organized by feature type (e.g.
mutations.csv, copyNum.csv, etc.). Each .CSV file should have the same # of rows, as the rows indicate the cell line.

- A series of "gene_list" .CSV files. These are the features which are in all of the feature files, which you believe
may be an important predictor for a particular result.

- A .CSV file for the results. These results will act as the outcome data (Y-values) for the machine learning model.
It should be either classification data (0, 1, etc.), or regression data. The # of rows in this file should be
equivalent to the number of rows in the features files.

- An arguments.txt file. This should be a plain text file which will detail various parameters including the name
of the results file, which machine learning models to create, etc.

All of these files should be in the same folder. This program will accept one argument, a path to that folder (it can
either be passed as a parameter on the command line, or entered at the prompt after running it). The outputs will be
.CSV files called RandomForestAnalysis.csv, LinearSVMAnalysis.csv, RadialBasisFunctionSVMAnalysis.csv,
ElasticNetAnalysis.csv, RidgeRegressionAnalysis.csv or/and LassoRegressionAnalysis depending on which machine learning modules you opt to use.

This program will also generate a HTML document summarizing how well each analysis performed and giving an overview
of the data submitted. Please use Google Chrome in order to see quantitative information when hovering over the respective
plots.

Also part of this program is the option to generate a series of .CSVs from MATLAB files.

## Table of Contents
Running the Cell Line Analyzer API involves three steps: <br />
1.) Dataset Formatting <br />
2.) The Arguments File <br />
3.) Running the Code <br />
4.) File Conversion <br />
5.) Summary Report <br />
6.) Troubleshooting <br />

# 1.) Dataset Formatting
### Dataset Formatting for the feature files:

Your feature .CSV files need to be formatted to the following specifications:

- Each one should have the same number of rows, where each row represents a single cell line.

- The only exception is the first row, which contains a label for each feature.

- Here's an example of a feature file with 3 features and 2 cell lines: <br />
```
feature1, feature2, feature3
1,0,.5
0,0,1.2
```

Each column should have the same type of data (e.g. floats, integers, or strings). Strings will be considered
categorical data and will be one-hot encoded. One feature file can have multiple types of data.

### Dataset Formatting for the results file:

The results.csv file is is a two column .CSV where the first column is simply the name of the cell line, and the
second column is the result for the cell line in the same row. This result should be an integer (0, 1, 2, etc. as
multiclassifiers are supported), or a float (for regression analysis). The number of rows should be equivalent to the
number of cell lines in all of your feature .CSV files.

- Here's an example of a results.csv file for a regressor with 2 cell lines: <br />
```
cell_line_name,result
cell_line0,0.46
cell_line1,0.32
```

### Dataset Formatting for the gene list files:
Also in the path should be .CSV files marked with the substring "gene_list" in the title, e.g. gene_list1.csv,
gene_list2.csv, etc. Please be aware of the gene names (to be of the same type as in the data, standard is the HUGO notation).

The format of each of these "gene list" files should be a simple list of genes found as features in the other feature
files, such as: <br />
```
PTEN, BRAF, TP53
```
across the first row of a csv.


# 2.) The Arguments File
### Specifying Parameters in `arguments.txt`

The arguments.txt file is simply a list of important, tunable parameters for this analysis. Some are required, and the
ones that aren't are marked with a star (*).

`results`: File Name of CSV

`data_split`*: Optional float between 0 and 1, representing how much data should be held out for each Monte Carlo
              subsampling. Defaults to 0.8.

`outer_monte_carlo_permutations`*: Integer representing the number of Monte Carlo outer subsamples to do for
                                   hyperparameter optimization. Defaults to 10.

`inner_monte_carlo_permutations`*: Integer representing the number of Monte Carlo inner subsamples to do for
                                   hyperparameter optimization. Defaults to 10.

`is_classifier`: 0 for regression, 1 for classification

`num_threads`* : Optional integer representing the number of threads to use for multi-processing operations. Defaults to
                 the number of CPUs on your computer. Cannot exceed this value. Be aware of this line if submitting jobs on
                 a cluster. Usually, the amount of threads has to be quantified on a cluster but without setting this integer
                 equal to the quantified one, the software will use all available resources what probably will
                 result in complications on the cluster.

`RandomForestAnalysis`*: The configuration for the Random Forest analysis. Accepts three parameters: True or False,
                         for whether the analysis should run at all. The number of outer monte carlo loops, and the
                         number of inner monte carlo loops. If not specified, defaults to True and the inner/outer
                         monte carlo param values set by other arguments in this file.

`LinearSVMAnalysis`: The configuration for the Linear SVM analysis. Accepts three parameters: True or False,
                     for whether the analysis should run at all. The number of outer monte carlo loops, and the
                     number of inner monte carlo loops. If not specified, defaults to True and the inner/outer
                     monte carlo param values set by other arguments in this file.

`RadialBasisFunctionSVMAnalysis`: The configuration for the Radial Basis Function SVM analysis. Accepts three
                                  parameters: True or False, for whether the analysis should run at all. The number of
                                  outer monte carlo loops, and the number of inner monte carlo loops. If not specified,
                                  defaults to True and the inner/outer monte carlo param values set by other arguments
                                  in this file.

`ElasticNetAnalysis`: The configuration for the ElasticNet analysis. Accepts three
                      parameters: True or False, for whether the analysis should run at all. The number of
                      outer monte carlo loops, and the number of inner monte carlo loops. If not specified,
                      defaults to True and the inner/outer monte carlo param values set by other arguments
                      in this file.

`RidgeRegressionAnalysis`: The configuration for the Ridge Regression analysis. Accepts three
                            parameters: True or False, for whether the analysis should run at all. The number of
                            outer monte carlo loops, and the number of inner monte carlo loops. If not specified,
                            defaults to True and the inner/outer monte carlo param values set by other arguments
                            in this file.

`LassoRegressionAnalysis`: The configuration for the Lasso Regression analysis. Accepts three
                            parameters: True or False, for whether the analysis should run at all. The number of
                            outer monte carlo loops, and the number of inner monte carlo loops. If not specified,
                            defaults to True and the inner/outer monte carlo param values set by other arguments
                            in this file.


`RandomSubsetElasticNetAnalysis`: The configuration for the Random Subset Elastic Net analysis. Accepts three
                                   parameters: True or False, for whether the analysis should run at all. The number of
                                   outer monte carlo loops, and the number of inner monte carlo loops. If not specified,
                                   defaults to True and the inner/outer monte carlo param values set by other arguments
                                   in this file.

`record_diagnostics`*: Optionally print out a diagnostics file which tells you which genes (and their indices) from
                       your gene_lists are missing in which feature files. Also, logs when optimal hyperparameter for a
                       particular training algorithm is on upper or lower bound of hyperparameter set.
                       Defaults to False.

`individual_train_algorithm`*: Optionally train only a specified ML algorithm with specified gene list feature combo and
                               specified hyperparameters. Acceptable parameters are:
                                    "RandomForestAnalysis"
                                    "LinearSVMAnalysis"
                                    "RadialBasisFunctionSVMAnalysis"
                                    "ElasticNetAnalysis"
                                    "RidgeRegressionAnalysis"
                                    "LassoRegressionAnalysis"

`individual_train_hyperparams`*: Optionally train only the specified ML algorithm with these hyperparams and a specified
                                 gene list feature combo. Values should be a comma separated list with no quotes or
                                 spaces.

`individual_train_combo`*: Optionally train only on the specified ML algorithm with this gene list feature combo and the
                           specified hyperparameters. Should be formatted as FEATURE_FILE_NAME:GENE_LIST_FILE_NAME
                           without the .csv extensions. See example below.

`rsen_combine_gene_lists`*: Optionally combine all gene lists into one when when selecting binary categorical features
                            for RandomSubsetElasticNet analysis.

`rsen_p_val`*: Optionally set the "p" value for RandomSubsetElasticNet analysis. Determines how predictions are scored.

Any .csv file in this path that is not the a gene list file, or the results.csv file, will be interpreted as a features
file.

### Example of completed argument.txt with proper syntax: 

```
results=results.csv
is_classifier=1
inner_monte_carlo_permutations=5
outer_monte_carlo_permutations=20
data_split=0.8
RandomForestAnalysis=True,10,5
```

### Example of a completed argument.txt with proper syntax for individual ML algorithm training:
```
results=results.csv
is_classifier=0
outer_monte_carlo_permutations=20
data_split=0.8
individual_train_algorithm=ElasticNetAnalysis
individual_train_combo=features_1int:significant_gene_list
individual_train_hyperparams=0.1,0.1
```


### Example of all files in the directory:

/SampleClassifierDataFolder...
   arguments.txt
   features_1.csv
   features_2.csv
   features_3.csv
   features_4.csv
   features_5.csv
   gene_list1.csv
   gene_list2.csv
   gene_list3.csv
   results.csv


# 3.) Running the Code
From the command line, type:

```
python __main__.py
```

Enter `0` for Analysis of Cell Lines

Type in the path of your desired folder, which contains `Arguments.txt`, Feature Data, and Output Data

Your results will be printed in the terminal and saved to either a RandomForestAnalysis.csv, a
LinearSVMAnalysis.csv, a RadialBasisFunctionSVMAnalysis.csv, a RidgeRegressionAnalysis.csv, a LassoRegressionAnalysis.csv
or/and an ElasticNetAnalysis.csv file. These files will be written to the directory from
which the program is called.

Alternatively, you can input the path of your target folder as an argument to the program:

```
python __main__.py 0 /PATH/TO/DATA_FOLDER
```

# 4.) File Conversion
This program also comes with a MATLAB to CSV file conversion tool. Since a lot of genomic data is often stored as
MATLAB files, this option allows you to take the contents of those MATLAB files and create a series of CSVs, one for
each declared variable. They will be created in the directory from which the program is called and all of the names will
be prepended with "convertedFile_". From there, there should be minimal tweaking of these files to get it in a format
which will be accepted by this data analysis.

To convert files, from the command line type:
```
python __main__.py
```
Enter `1` to convert the file and type in the path of the .mat file.

Alternatively, you can input the path of your .mat file as an argument to the program:

```
python __main__.py 1 /PATH/TO/.MAT_FILE
```
It will automatically distinguish the input. If it's a folder, it will attempt to run a cell line analysis, and if it's
a .mat file, it will attempt to convert it to a series of .CSVs.

# 5.) Summary Report
Once the analysis has finished, a summary report will be generated in the input folder. This report has several
components:
- An bar chart showing the score breakdown for the best performing algorithm. The score is determined by the average
score (R^2 for regressors or accuracy for classifiers) for all outer monte carlo runs - the standard deviation for all
of those runs. The best performing algorithm is selected by picking the one with the highest scoring gene list, feature
file combo.
- For each algorithm, the following three plots:
    - A histogram of the scores (mean R^2 or accuracy) for the top 5, bottom 5, and a randomly selected set
      of 15 gene list and feature file combos. Shown with averages and error bars.
    - A histogram of the accuracies (only for regression, where it is the mean squared errors) for the same 25 combos. Also
      shown with averages and error bars.
    - A pie chart of how well each gene list performs, surrounded by a donut chart of how well each feature file
      performs all relative to each other.


# 6.) Troubleshooting

- This program requires python 3.6. You can check your version of python by running typing `python --version` in the
  command line. If it's 3.6.x, please upgrade.

- Make sure that your is_classifier variable matches what is in your results.csv file. If your results.csv file is
  filled with 0s and 1s for the values, but is_classifier=0, we'll try to interpret all that data as regression data. At
  best, the results will be nonsensical, at worst, it will error when attempting to create a proper machine learning
  model.

- This program also requires installation of the following packages: scipy, numpy, pandas, joblib, sklearn
  Make sure all of these packages are installed for python 3.6. If you don't have them, you can install them with pip
  from the command line:
  `pip install PACKAGE_NAME` where PACKAGE_NAME is scipy, sklearn, etc.

- Please not for the accuracy score of classification: This measure is the accuracy score as provided by sklearn. This metric
  can perform very bad, depending on the base rates of the classes. If there are two classes ('sensitive' and 'resistant', for
  example) and one of these classes ('sensitive') has a low base rate, one gets a high accuracy score when always predicting
  'resistant'. Thus, it could be possible that a model with given accuracy score performs much better than one with a higher 
  accuracy score, which is known as the accuracy paradox. If another measure is whished, it should be manually coded.

- Be aware that LinearSVM needs less features than samples in order to work properly. Using gene lists, the amount of features
  is significantly reduced, but usually this amount is still larger than the amount of samples (cell lines). If you use two
  feature files (for instance, gene expression and mutation data sets) and gene lists with a total gene count of 500, you would
  get 1000 features as a result. Usually, the amount of cell lines is smaller than that. The same logic applies to linear 
  regression models (RidgeRegression, LassoRegression and ElasticNet), but the algorithms applied in the CLA have much tuning
  implemented so that some coefficients are set to zero.

The Python package installer pip should come standard on unix bases operating systems. For Windows, you'll need to use
Powershell to install both python and pip.


# Authorship

Cell Line Analyzer was created by Andrew Patti, Georgios Patoulidis and Christine Zhang under supervision of David Craft,
Massachusetts General Hospital - Harvard Medical School. 2018.
