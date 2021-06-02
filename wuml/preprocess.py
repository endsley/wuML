#!/usr/bin/env python

from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys
import os

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)



def center_and_scale(X):
	X = preprocessing.scale(X)
	return X

def center_scale_with_missing_data(X, replace_nan_with_0=False): 
	d = X.shape[1]
	ignore_column_with_0_σ = []
	for i in range(d):
		x = X[:,i]
		ẋ = x[np.invert(np.isnan(x))]
		ẋ = ẋ - np.mean(ẋ)
		σ = np.std(ẋ)

		if σ < 0.00001:
			ignore_column_with_0_σ.append(i)
		else:
			X[np.invert(np.isnan(x)), i] = ẋ/σ

	for i in ignore_column_with_0_σ:
		X = np.delete(X, i , axis=1)	# delete column with σ=0

	if replace_nan_with_0:
		X = np.nan_to_num(X)

	return X, ignore_column_with_0_σ


#def read_csv(filename, preprocess_list=[]):
#	df = pd.read_csv (filename, header='infer',index_col=0)
#	X = df.values
#
#	import pdb; pdb.set_trace()
#	#X = genfromtxt(filename, delimiter=',')
#
#	for f in preprocess_list:
#		X = f(X)
#
#	return X

def gen_10_fold_data(data_name, data_path='./data/'):

	xpath = data_path + data_name

	X = np.loadtxt(xpath + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt(xpath + '_label.csv', delimiter=',', dtype=np.int32)			

	fold_path = xpath + '/'
	if os.path.exists(fold_path): 
		pass
	else:
		os.mkdir(fold_path)


	#if os.path.exists(fold_path): 
	#	txt = input("There's already a 10 fold data for %s, generate a new set? (y/N)"%data_name)
	#	if txt == 'y':
	#		pass
	#	else:
	#		return
	#else:
	#	os.mkdir(fold_path)


	kf = KFold(n_splits=10, shuffle=True)
	kf.get_n_splits(X)
	loopObj = enumerate(kf.split(X))

	for count, data in loopObj:
		[train_index, test_index] = data

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '.csv', X_train, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_label.csv', Y_train, delimiter=',', fmt='%d') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_test.csv', X_test, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_label_test.csv', Y_test, delimiter=',', fmt='%d') 

