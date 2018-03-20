# CellLineAnalyzer

Running the Cell Line Analyzer API involves three steps:
1. Dataset Formatting
2. Specifying Parameters in `Arguments.txt`
3. Running the Code

## Dataset Formatting

Your datasets need to be formatted to the below specifications. 

_n_ = number of samples (CCLIDs) 

_m_p_ = number of features of type p (ex. _p_ = mutation)

### Y values (_n_ x 2):

Column 1: CCLIDs

Column 2: Y value (ex. radiation output)

Y values assumed to be numerical

### X values (_n_ x _m_p_): 

All X matrices must be of the same input type: Categorical or Numerical

Column header contains HUGO gene 

X Matrices named: [feature]_[input type].csv

## Arguments.txt File

To run Cell Line Analyzer, update the `arguments.txt` file. 

`results`: "File Name of CSV"

`data_split`: [x%, y%, z%], where x + y + z = 100

`is_classifier`: 0 for regression, 1 for classification

Also in the path should be files marked with the string "gene_list" in them, e.g. gene_list1.csv,
gene_list2.csv, etc.

The format of each of these "gene list" files should be a simple list of genes found as features in the other feature
files, such as:
WNT, ERK, p53, beta-catenin
across the first row of a csv.

Any .csv file in this path that is not the a gene list file, or the results.csv file, will be interpreted as a features
file.

### Example of completed argument.txt with proper syntax: 

```
results=results.csv
data_split=[80,10,10]
is_classifier=1
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

## Running the Code

```
python __main__.py
```

Enter `0` for Analysis of Cell Lines

Type in the path of your desired folder, which contains `Arguments.txt`, Feature Data, and Output Data

Your results will be printed in the terminal.

Alternatively, you can input the path of your target folder as an argument to the program:

```
python __main__.py /PATH/TO/DATA_FOLDER
```


## Authorship

Cell Line Analyzer was created by Andrew Patti and Christine Zhang under supervision of David Craft, MGH/HMS. 2018.
