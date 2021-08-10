#!/usr/bin/env python

from numpy import genfromtxt
from wuml.IO import *
from wuml.data_stats import *
from wuml.data_loading import *
from wuml.type_check import *

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

def remove_rows_with_too_much_missing_entries(dataFrame, threshold=0.6, newDataFramePath=''):
	'''
		If a row is kept if it has more than "threshold" percentage of normal data
		dataFrame is a pandas format, it can be converted this way
		pd.DataFrame(data=data[1:,1:],    # values
						index=data[1:,0],    # 1st column as index
						columns=data[0,1:])  # 1st row as the column names
	'''
	df = dataFrame.get_data_as('DataFrame')

	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	pth = './results/Preprocessing/'
	ensure_path_exists('./results')
	ensure_path_exists(pth)

	#	Obtain a decimated dataframe
	limitPer = int(d * threshold)
	df_decimated = df.dropna(thresh=limitPer, axis=0)
	oldID = df['id'].values.tolist()
	newID = df_decimated['id'].values.tolist()
	removed_samples = set(oldID).difference(newID)

	#	Record the results
	output_str = 'Original dataFrame dimension : %d samples,  %d dimensions\n'%(n, d)
	output_str += 'Deciated dataFrame dimension : %d samples,  %d dimensions\n'%(df_decimated.values.shape[0], df_decimated.values.shape[1])
	output_str += 'Removed Rows missing at least these percentage of entries : %.3f\n'%(1-threshold)

#	#	Record the removed features 
	output_str += '\nID of Removed Rows\n'
	output_str += str(removed_samples)

	write_to(output_str, pth + 'row_decimation_info.txt')
	if newDataFramePath != '': df_decimated.to_csv(path_or_buf=newDataFramePath, index=False)

	df_decimated = wData(dataFrame=df_decimated)
	return df_decimated


def remove_columns_with_too_much_missing_entries(dataFrame, threshold=0.6, newDataFramePath=''):
	'''
		If a column is kept if it has more than "threshold" percentage of normal data
		dataFrame is a pandas format, it can be converted this way
		pd.DataFrame(data=data[1:,1:],    # values
						index=data[1:,0],    # 1st column as index
						columns=data[0,1:])  # 1st row as the column names
	'''
	df = dataFrame.get_data_as('DataFrame')

	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	pth = './results/Preprocessing/'
	ensure_path_exists('./results')
	ensure_path_exists(pth)

	#	Obtain a decimated dataframe
	limitPer = int(n * threshold)
	df_decimated = df.dropna(thresh=limitPer, axis=1)
	oldColumns = df.columns.values.tolist()
	newColumns = df_decimated.columns.values.tolist()
	removed_columns = set(oldColumns).difference(newColumns)

	#	Record the results
	output_str = 'Original dataFrame dimension : %d samples,  %d dimensions\n'%(n, d)
	output_str += 'Deciated dataFrame dimension : %d samples,  %d dimensions\n'%(df_decimated.values.shape[0], df_decimated.values.shape[1])
	output_str += 'Removed Columns missing at least these percentage of entries : %.3f\n'%(1-threshold)

	#	Record the removed features 
	output_str += '\nRemoved Columns + Missing Percentage'
	max_width = len(max(removed_columns, key=len))
	for column in removed_columns:
		x = df[column].values
		missing_percentage = np.sum(np.isnan(x))/n
		output_str += (('\n\t%-' + str(max_width) + 's\t%.2f')%(column, missing_percentage))

	#	Record the retained features 
	output_str += '\n\nRetained Columns + Missing Percentage'
	for column in df_decimated:
		x = df_decimated[column].values
		missing_percentage = np.sum(np.isnan(x))/n
		output_str += (('\n\t%-' + str(max_width) + 's\t%.2f')%(column, missing_percentage))

	write_to(output_str, pth + 'column_decimation_info.txt')
	if newDataFramePath != '': df_decimated.to_csv(path_or_buf=newDataFramePath, index=False)

	df_decimated = wData(dataFrame=df_decimated)
	return df_decimated

def decimate_data_with_missing_entries(dataFrame, column_threshold=0.6, row_threshold=0.6,newDataFramePath=''):
	'''
		It will automatically remove rows and columns of a dataFrame with missing entries.
	'''
	df = dataFrame.get_data_as('DataFrame')
	dfSize = 'Data size:' + str(df.shape)

	mdp = np.array(identify_missing_data_per_feature(dataFrame))
	x = np.arange(1, len(mdp)+1)

	lp = lines(figsize=(10,5))
	lp.plot_line(x, mdp, 'Before Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=dfSize, subplot=121, ylim=[0,1], xTextLoc=0, yTextLoc=0.9)

	df = remove_columns_with_too_much_missing_entries(dataFrame, threshold=column_threshold)
	df_decimated = remove_rows_with_too_much_missing_entries(df, threshold=row_threshold, newDataFramePath=newDataFramePath)
	dfSize = 'Data size:' + str(df_decimated.shape)

	mdp = np.array(identify_missing_data_per_feature(df_decimated))
	x = np.arange(1, len(mdp)+1)

	lp.plot_line(x, mdp, 'After Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=dfSize, subplot=122, ylim=[0,1], xTextLoc=0, yTextLoc=0.9)
	lp.show()

	return df_decimated


def center_and_scale(wuData):
	X = wuData.get_data_as('ndarray')
	X = preprocessing.scale(X)
	wuData.df = pd.DataFrame(X)
	return wuData



def center_scale_with_missing_data(X, replace_nan_with_0=False): 
	'''
		For each column, find μ, σ while ignoring the entries that are zero. 
	'''
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

def rearrange_sample_to_same_class(X,Y):
	l = np.unique(Y)
	newX = np.empty((0, X.shape[1]))
	newY = np.empty((0))

	for i in l:
		indices = np.where(Y == i)[0]
		newX = np.vstack((newX, X[indices, :]))
		newY = np.hstack((newY, Y[indices]))

	return [newX, newY]


#	Relating all samples to the most likely one, with p(X1) > p(Xi) for all i
#	Given X1 X2 with p(X1)/p(X2)=2  the weight for X1 = 1, and X2 = 2
def get_likelihood_weight(data):
	X = ensure_numpy(data)
	
	Pₓ = wuml.KDE(X)
	logLike = Pₓ(X, return_log_likelihood=True)
	max_likely = np.max(logLike)
	ratios = np.exp(max_likely - logLike)
	return wuml.wData(X_npArray=ratios)

