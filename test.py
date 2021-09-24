#!/usr/bin/env python

import wuml 

'''
	This code loads data with missing entries at random as a wData type
	It will automatically remove the features and samples that are missing too many entries
	On the decimated data, it will perform imputation
	It will lastly save and export the results to a csv file
'''

wuml.set_terminal_print_options(precision=3)
#data = wuml.wData('../../data/chem.exposures.csv', row_id_with_label=0)
data = wuml.wData('examples/data/missin_example.csv', row_id_with_label=0)
#data.delete_column('Unnamed: 0')	# remove multiple unnecessary columns by name

print(data)


#	column_threshold=0.95, this will keep features that are at least 95% full
#	Note that an id column is "required" for this process to document which rows are removes
dataDecimated = wuml.decimate_data_with_missing_entries(data, column_threshold=0.50, row_threshold=0.70,newDataFramePath='')

print(dataDecimated)

X = wuml.impute(dataDecimated, ignore_first_index_column=True)		# perform mice imputation
#X.to_csv('../../data/Chem_decimated_imputed.csv')	

print(dataDecimated)







##!/usr/bin/env python
#
#import wuml
#
#
###	We generated a synthetic data for regression with 4 dimensions where
###	x1 x2 has positive influence
###	x3 has no influence
###	x4 has negative influence
##
#
#data = wuml.wData(xpath='examples/data/shap_regress_example.csv', batch_size=20, 
#					label_type='continuous', label_column_name='label', 
#					row_id_with_label=0)
#
#EXP = wuml.explainer(data, load_network_path='./shap_test_network.pk')
#
## Show the regression results
#Ŷ = EXP.net(data, output_type='ndarray')
#SR_train = wuml.summarize_regression_result(data.Y, Ŷ)
#O = SR_train.true_vs_predict()
#print(O)
#
## Show the explanation results
#explanation = EXP(data)	# outputs the weight importance
#print(explanation)


