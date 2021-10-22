#!/usr/bin/env python

import wuml 

'''
	This code loads data with missing entries at random as a wData type
	It will automatically remove the features and samples that are missing too many entries
	On the decimated data, it will perform imputation
	It will lastly save and export the results to a csv file
'''

wuml.set_terminal_print_options(precision=3)
data = wuml.wData('examples/data/missin_example.csv', row_id_with_label=0)
print(data)

dataDecimated = wuml.decimate_data_with_missing_entries(data, column_threshold=0.50, row_threshold=0.70,newDataFramePath='')
print(dataDecimated)

X = wuml.impute(dataDecimated, ignore_first_index_column=True)		# perform mice imputation

X.plot_2_columns_as_scatter('B','C')
import pdb; pdb.set_trace()
