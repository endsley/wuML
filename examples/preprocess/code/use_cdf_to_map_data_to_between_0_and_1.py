#!/usr/bin/env python

import wuml 

#	We first remove column A because there are too many missing values
data = wuml.wData('../../data/missin_example.csv', row_id_with_label=0, columns_to_ignore=['A'])
print('We start with this dataset')
print(data,'\n')

X = wuml.impute(data, ignore_first_index_column=False)
print('We first impute all the residual missing entries')
print(X,'\n')


X = wuml.map_data_between_0_and_1(X, map_type='linear') # map_type: linear, or cdf
print(X[0:20,:])


X = wuml.map_data_between_0_and_1(X, map_type='cdf') # map_type: linear, or cdf
print(X[0:20,:])

