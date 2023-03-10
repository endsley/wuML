#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np
import torch
import torch.nn as nn

wuml.set_terminal_print_options(precision=3)





# 	Here, we load the data, the network, and show that the explainer is embedded within the model
#	This ability currently only works on combined network class, and not simpler classes
data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					extra_data='../../data/wine_regress_label.csv', 
					preprocess_data='center and scale', 
					 batch_size=16, label_type='discrete')
Y2 = data.extra_data_dictionary['numpy'][0]

model = wuml.load_torch_network('combined_net.pk', load_as_cpu_or_gpu='cpu')
ŷ = model(data)
exp = model.explainer(data, y=Y2)


