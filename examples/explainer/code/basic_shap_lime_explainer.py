#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np


#	regression test
#	data is designed such that 	x₁, x₂ has minor positive impact x₄ has major negative, 
#		5x₁ + x₂ + x₁x₂ - 8x₄ - 2x₄x₄ + δ
data = wuml.wData('../../data/shap_regress_example_gaussian.csv', first_row_is_label=True, 
					label_type='continuout', label_column_name='label', preprocess_data='center and scale')

model = wuml.regression(data, regressor='linear')
E = wuml.explainer(data, model, explainer_algorithm='shap')
exp = E(data)



#	classification test
data = wuml.wData('../../data/shap_classifier_example.csv', first_row_is_label=True, 
					label_type='discrete', label_column_name='label')

model = wuml.classification(data, classifier='LogisticRegression')
E = wuml.explainer(data, model, explainer_algorithm='lime')
exp = E(data)

