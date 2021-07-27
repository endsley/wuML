#!/usr/bin/env python

import wuml
import pandas as pd

## Show featurehistogram after data decimation
## 	header : the feature name you want to use to see the histogram. 
## 	index_col : if 0, it means the 1st row of dataFrame is the feature names. None would be nothing

df = pd.read_csv ('../data/exposures_row_decimated.csv', header='infer',index_col=0)
wuml.get_feature_histograms(df['finalga_best'].values, title='Histogram of Data Labels')


