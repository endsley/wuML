#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
import wplotlib



data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', label_type='discrete')
sampled_data = wuml.get_N_samples_from_each_class(data, 5)
wuml.jupyter_print(sampled_data)
wuml.jupyter_print(sampled_data.Y)
