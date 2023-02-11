#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 

'''
	This code loads data with missing entries at random as a wData type
	It will automatically remove the features and samples that are missing too many entries
	On the decimated data, it will perform imputation
	It will lastly save and export the results to a csv file
'''

wuml.set_terminal_print_options(precision=3)
data = wuml.wData('../../data/missin_example.csv', first_row_is_label=True)
wuml.jupyter_print(data)

#	column_threshold=0.95, this will keep features that are at least 95% full
#	Note that an id column is "required" for this process to document which rows are removes
dataDecimated = wuml.decimate_data_with_missing_entries(data, column_threshold=0.50, row_threshold=0.70,newDataFramePath='')
wuml.jupyter_print(dataDecimated)

X = wuml.impute(dataDecimated, ignore_first_index_column=True)		# perform mice imputation
wuml.jupyter_print(X)

