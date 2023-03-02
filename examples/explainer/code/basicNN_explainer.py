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


model = wuml.basicNetwork('mse', data, networkStructure=[(30,'relu'),(50,'relu'),(1,'none')], max_epoch=30)
model.train(print_status=True)
model.output_y_as = 'not column'
y = model(data)

E = wuml.explainer(data, model, explainer_algorithm='shap')
exp = E(data)

E = wuml.explainer(data, model, explainer_algorithm='lime')
exp = E(data)

