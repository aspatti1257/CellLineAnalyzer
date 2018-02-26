# CellLineAnalyzer

## Dataset Format

Your datasets need to be formatted to the below specifications. 

n = number of samples (CCLIDs) 

m_p = number of features of type p (ex. p = mutation)

### Y values:
n x 2

Column 1: CCLIDs

Column 2: Y value (ex. radiation output)

Y values assumed to be numerical

### X values: 
n x m_p

All X matrices must be of the same type

Column header contains HUGO gene 

X Matrices named:

[feature]_[input type].csv

Input types: Numerical, Categorical 

## Arguments.txt File

To run Cell Line Analyzer, update the arguments.txt file. 

results="File Name of CSV"

data_split=[x%, y%, z%], where x + y + z = 100

important_features=Comma separated list of file.feature names

is_classifier=[1 for classification, 0 for regression]

### Example of completed argument.txt with proper syntax: 

```
results=results.csv
data_split=[80,10,10]
important_features=features.feature_two, categorical.feature_cat
is_classifier=1
```

## Running the Code

```
python __main__.py
```

Enter '0' for Analysis of Cell LInes

Type in the path of your desired folder, which contains Arguments.txt, Feature Data, and Output Data

Your results will be printed in the terminal 
