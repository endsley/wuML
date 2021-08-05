#!/usr/bin/env python

import wuml 

'''
	This code loads data with missing entries at random as a wData type
	It will automatically remove the features and samples that are missing too many entries
	On the decimated data, it will perform imputation
	It will lastly save and export the results to a csv file
'''

data = wuml.wData('examples/data/chem.exposures.csv', row_id_with_label=0)
dataDecimated = wuml.decimate_data_with_missing_entries(data, column_threshold=0.95, row_threshold=0.9,newDataFramePath='')
#	column_threshold=0.95, this will keep features that are at least 95% full


X = wuml.impute(dataDecimated)		# perform mice imputation
X.to_csv('examples/data/Chem_decimated_imputed.csv')	


