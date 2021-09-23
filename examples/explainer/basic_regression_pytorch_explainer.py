#!/usr/bin/env python

import wuml
import sklearn
import shap
import wplotlib
import numpy as np
from wplotlib import histograms
import torch
import torch.nn as nn
from torch.autograd import Variable


##	We generated a synthetic data for regression with 4 dimensions where
##	x1 x2 has positive influence
##	x3 has no influence
##	x4 has negative influence
#

data = wuml.wData(xpath='../data/shap_regress_example.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)


EXP = wuml.explainer(data, 	loss='mse',		# This will create a network for regression and explain instance wise 
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=150, learning_rate=0.001)

explanation = EXP(data)	# outputs the weight importance
print(explanation)

