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


data = wuml.wData(xpath='examples/data/shap_regress_example.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)

if wuml.path_exists('./shap_test_network.pk'):
	bNet = wuml.pickle_load('./shap_test_network.pk')
else:
	bNet = wuml.basicNetwork('mse', data, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=200, learning_rate=0.001)
	bNet.train()
	bNet.eval(output_type='ndarray')
	
	Ŷ = bNet(data, output_type='ndarray')
	SR_train = wuml.summarize_regression_result(data.Y, Ŷ)
	O = SR_train.true_vs_predict()
	print(O)
	wuml.pickle_dump(bNet, './shap_test_network.pk')



explainer = shap.KernelExplainer(bNet, data.X, link="logit")
shap_values = explainer.shap_values(data[0,:], nsamples=20)
import pdb; pdb.set_trace()


