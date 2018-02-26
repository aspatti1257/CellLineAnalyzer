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

`important_features`: Comma separated list of file.feature names

`is_classifier`: 0 for regression, 1 for classification

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

Enter `0` for Analysis of Cell Lines

Type in the path of your desired folder, which contains `Arguments.txt`, Feature Data, and Output Data

Your results will be printed in the terminal 

## Authorship

Cell Line Analyzer was created by Andrew Patti and Christine Zhang under supervision of David Craft, MGH/HMS. 2018.
