#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import numpy as np
import wuml

n = 2000

#	Sine Data
dat_x = 9.3*np.random.rand(n,1)
dat_y = np.sin(dat_x)
sine_data = np.hstack((dat_x,dat_y)) + 0.06*np.random.randn(n,2)

sine_data_100 = wuml.sampling_rows_from_matrix(100, sine_data)
sine_data_200 = wuml.sampling_rows_from_matrix(200, sine_data)
sine_data_500 = wuml.sampling_rows_from_matrix(500, sine_data)
sine_data_1000 = wuml.sampling_rows_from_matrix(1000, sine_data)
sine_data_1500 = wuml.sampling_rows_from_matrix(1500, sine_data)



sine_hsic = wuml.HSIC(sine_data[:,0], sine_data[:,1], sigma_type='opt', normalize_hsic=False)
sine_hsic_100 = wuml.HSIC(sine_data_100[:,0], sine_data_100[:,1], sigma_type='opt', normalize_hsic=False)
sine_hsic_200 = wuml.HSIC(sine_data_200[:,0], sine_data_200[:,1], sigma_type='opt', normalize_hsic=False)
sine_hsic_500 = wuml.HSIC(sine_data_500[:,0], sine_data_500[:,1], sigma_type='opt', normalize_hsic=False)
sine_hsic_1000 = wuml.HSIC(sine_data_1000[:,0], sine_data_1000[:,1], sigma_type='opt', normalize_hsic=False)
sine_hsic_1500 = wuml.HSIC(sine_data_1500[:,0], sine_data_1500[:,1], sigma_type='opt', normalize_hsic=False)


print('\tHSIC full size     : ', sine_hsic, ', size:', sine_data.shape)
print('\tHSIC approximation : ', sine_hsic_100, ', size:', sine_data_100.shape)
print('\tHSIC approximation : ', sine_hsic_200, ', size:', sine_data_200.shape)
print('\tHSIC approximation : ', sine_hsic_500, ', size:', sine_data_500.shape)
print('\tHSIC approximation : ', sine_hsic_1000, ', size:', sine_data_1000.shape)
print('\tHSIC approximation : ', sine_hsic_1500, ', size:', sine_data_1500.shape)
