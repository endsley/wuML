#!/usr/bin/env python

import wuml 

#	We first remove column A because there are too many missing values
data = wuml.wData('examples/data/missin_example.csv', row_id_with_label=0, columns_to_ignore=['A'])
print('We start with this dataset')
print(data,'\n')

X = wuml.impute(data, ignore_first_index_column=False)
print('We first impute all the residual missing entries')
print(X,'\n')


X = wuml.use_reverse_cdf_to_map_data_between_0_and_1(X)
print('We lastly use reverse CDF to map the data to values between 0 and 1')
print(X,'\n')

