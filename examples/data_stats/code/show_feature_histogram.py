#!/usr/bin/env python

import wuml

## Show featurehistogram after data decimation
## 	header : the feature name you want to use to see the histogram. 
## 	index_col : if 0, it means the 1st row of dataFrame is the feature names. None would be nothing
data = wuml.wData('../../data/Chem_decimated_imputed.csv', row_id_with_label=0)


## You can pick a column from the data using its column name, here it is 'finalga_best'
wuml.get_feature_histograms(data['finalga_best'], title='Histogram of Data Labels', ylogScale=False)
wuml.get_feature_histograms(data['finalga_best'], title='Histogram of Data Labels', ylogScale=True)


