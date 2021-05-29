#!/usr/bin/env python

from numpy import genfromtxt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys

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


def read_csv(filename, preprocess_list=[]):
	df = pd.read_csv (filename, header='infer',index_col=0)
	X = df.values

	import pdb; pdb.set_trace()
	#X = genfromtxt(filename, delimiter=',')

	for f in preprocess_list:
		X = f(X)

	return X
