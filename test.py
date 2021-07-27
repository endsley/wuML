#!/usr/bin/env python

import wuml 
#import numpy as np
#import pandas as pd
#import torch
import sys
print(sys.path)


#	Preprocess examples
#import examples.preprocess.decimate_with_missing_data 


#	We prefer dataFrame where 1st row have feature labels and the 1st column consists of sample id
#df = pd.read_csv ('./data/wine.csv', header=None)
#df = pd.read_csv ('./data/exposures_row_decimated.csv', header='infer',index_col=0)

## Decimate DataFrame with missing data
#data = wuml.load_csv('./data/chem.exposures.csv', row_id_with_label=0)
#wuml.missing_data_stats(data)


## Decimate DataFrame with missing data
#data = wuml.load_csv('./data/chem.exposures.csv', row_id_with_label=0)
#dfSmall = wuml.decimate_data_with_missing_entries(data, column_threshold=0.7, row_threshold=0.7,newDataFramePath='')


## Show label histogram after data decimation
#df = pd.read_csv ('./data/exposures_row_decimated.csv', header='infer',index_col=0)
#wuml.get_feature_histograms(df['finalga_best'].values, title='Histogram of Data Labels')


###	Test out the built-in neural network
#X = wuml.load_csv('./data/regress.csv', './data/regress_label.csv',row_id_with_label=None)
##X = wuml.load_csv('./data/wine.csv', './data/wine_label.csv',row_id_with_label=None)
##X = wuml.center_and_scale(X)
##X = wuml.load_csv('./data/chem.exposures.csv', row_id_with_label=0)
#def costFunction(x, y, ŷ, ind):
#	ŷ = torch.squeeze(ŷ)
#	return torch.sum((y- ŷ) ** 2)	
#
#bNet = wuml.basicNetwork(costFunction, X, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=3000, learning_rate=0.001)
#bNet.train()

#foo = [wPr.center_and_scale]
#X = wPr.read_csv('./data/chem.exposures.csv', preprocess_list=foo)
#X = wPr.read_csv('./data/wine.csv', preprocess_list=foo)


