#!/usr/bin/env python

import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
import numpy as np

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='center and scale')

SC = wuml.clustering(data, n_clusters=3, method='Spectral Clustering', gamma=0.05)
SC.nmi_against_true_label()
SC.plot_scatter_result()
SC.save_clustering_result_to_csv('./clusters.csv')
