#!/usr/bin/env python

import wuml


##	We generated a synthetic data for regression with 4 dimensions where
##	x1 x2 has positive influence
##	x3 has no influence
##	x4 has negative influence
#

data = wuml.wData(xpath='../../data/shap_regress_example.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)

EXP = wuml.explainer(data, load_network_path='./shap_test_network.pk')	# the network should be a basicNetwork type

# Show the regression results
Ŷ = EXP.model(data, output_type='ndarray')
SR_train = wuml.summarize_regression_result(data.Y, Ŷ)
O = SR_train.true_vs_predict()
print(O)

# Show the explanation results
explanation = EXP(data)	# outputs the weight importance
print(explanation)


