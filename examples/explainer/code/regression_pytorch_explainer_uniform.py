#!/usr/bin/env python
import wuml


##	We generated a synthetic data for regression with 4 dimensions where
##	x1 x2 has positive influence
##	x3 has no influence
##	x4 has negative influence
#

data = wuml.wData(xpath='../../data/shap_regress_example_uniform.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)


EXP = wuml.explainer(data, 	loss='mse',		# This will create a network for regression and explain instance wise 
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=150, learning_rate=0.001, print_network_training_status=False)


# Show the regression results
Ŷ = EXP.net(data, output_type='ndarray')
SR_train = wuml.summarize_regression_result(data.Y, Ŷ)
print(SR_train.true_vs_predict())

# Show the explanation results
explanation = EXP(data)	# outputs the weight importance
print(explanation)


# save the result
wuml.pickle_dump(EXP.net, './shap_test_network.pk')


