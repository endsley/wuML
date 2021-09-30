#!/usr/bin/env python
import wuml


##	We generated a synthetic data for regression with 4 dimensions where
##	x1 x2 has positive influence
##	x3 has no influence
##	x4 has negative influence
##
##	x1 has normal distribution
##	x2 is exponential distribution but minus 2 so it could be negative
##	x3 is uniform but shouldn't matter
##	x4 is categorical distribution.

data = wuml.wData(xpath='../../data/shap_regress_example_mix_distributions.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', row_id_with_label=0)



#	Example 1
EXP = wuml.explainer(data, 	loss='mse',		# This will create a network for regression and explain instance wise 
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=150, learning_rate=0.001, print_network_training_status=True)

# Show the explanation results
explanation = EXP(data)	# outputs the weight importance
print('Notice that since x1 and x2 can be negative, we get both negative and positive influence.')
print('This is not correct since they should only have positive influence.')
print(explanation)



#	Example 2
Cdata = wuml.center_and_scale(data)
EXP2 = wuml.explainer(Cdata, 	loss='mse',		# This will create a network for regression and explain instance wise 
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=150, learning_rate=0.001, print_network_training_status=True)

# Show the explanation results
explanation = EXP2(Cdata)	# outputs the weight importance
print('Notice that since x1, x2, x4 can be both positive and negative, which is not correct.')
print(explanation)


#	Example 3
Udata = wuml.use_cdf_to_map_data_between_0_and_1(data, output_type_name='wData')
EXP3 = wuml.explainer(Udata, 	loss='mse',		# This will create a network for regression and explain instance wise 
						networkStructure=[(600,'relu'),(600,'relu'),(600,'relu'),(1,'none')], 
						max_epoch=600, learning_rate=0.001, print_network_training_status=True)

# Show the regression results
Ŷ = EXP3.model(Udata, output_type='ndarray')
SR_train = wuml.summarize_regression_result(Udata.Y, Ŷ)
print(SR_train.true_vs_predict())

# Show the explanation results
explanation = EXP3(Udata)	# outputs the weight importance
print('If we map all the data into the range of [0,1], notice that x1, x2, x4 all have the correct attribution sign')
print('We noticed that it requires a larger network, and longer training time, but it gives better attributions.')
print(explanation)


