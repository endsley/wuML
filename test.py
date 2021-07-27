#!/usr/bin/env python

import wuml 
import torch

#import numpy as np
#import pandas as pd
#import sys



##	Test out the built-in neural network
X = wuml.load_csv('examples/data/regress.csv', 'examples/data/regress_label.csv',row_id_with_label=None)
#X = wuml.load_csv('./data/wine.csv', './data/wine_label.csv',row_id_with_label=None)
#X = wuml.center_and_scale(X)
#X = wuml.load_csv('./data/chem.exposures.csv', row_id_with_label=0)
def costFunction(x, y, ŷ, ind):
	ŷ = torch.squeeze(ŷ)
	return torch.sum((y- ŷ) ** 2)	

bNet = wuml.basicNetwork(costFunction, X, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=3000, learning_rate=0.001)
bNet.train()

#foo = [wPr.center_and_scale]
#X = wPr.read_csv('./data/chem.exposures.csv', preprocess_list=foo)
#X = wPr.read_csv('./data/wine.csv', preprocess_list=foo)


