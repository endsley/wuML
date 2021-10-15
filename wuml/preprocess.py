#!/usr/bin/env python

from wuml.IO import *
from wuml.data_stats import *
from wuml.data_loading import *
from wuml.type_check import *

from sklearn.model_selection import train_test_split
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

def remove_rows_with_too_much_missing_entries(data, threshold=0.6, newDataFramePath=''):
	'''
		If a row is kept if it has more than "threshold" percentage of normal data
		data is a pandas format, it can be converted this way
		pd.DataFrame(data=data[1:,1:],    # values
						index=data[1:,0],    # 1st column as index
						columns=data[0,1:])  # 1st row as the column names
	'''
	df = wuml.ensure_DataFrame(data)

	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	pth = './results/'
	ensure_path_exists('./results')
	#ensure_path_exists(pth)

	#	Obtain a decimated dataframe
	limitPer = int(d * threshold)
	df_decimated = df.dropna(thresh=limitPer, axis=0)
	oldID = df.index.values.tolist()
	newID = df_decimated.index.values.tolist()
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


def remove_columns_with_too_much_missing_entries(data, threshold=0.6, newDataFramePath=''):
	'''
		If a column is kept if it has more than "threshold" percentage of normal data
		data is a pandas format, it can be converted this way
		pd.DataFrame(data=data[1:,1:],    # values
						index=data[1:,0],    # 1st column as index
						columns=data[0,1:])  # 1st row as the column names
	'''
	df = wuml.ensure_DataFrame(data)

	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	pth = './results/'
	ensure_path_exists('./results')
	#ensure_path_exists(pth)

	#	Obtain a decimated dataframe
	limitPer = int(n * threshold)
	df_decimated = df.dropna(thresh=limitPer, axis=1)
	oldColumns = df.columns.values.tolist()
	newColumns = df_decimated.columns.values.tolist()
	removed_columns = set(oldColumns).difference(newColumns)

	#	Record the results
	output_str = 'Original data dimension : %d samples,  %d dimensions\n'%(n, d)
	output_str += 'Deciated data dimension : %d samples,  %d dimensions\n'%(df_decimated.values.shape[0], df_decimated.values.shape[1])
	output_str += 'Removed Columns missing at least these percentage of entries : %.3f\n'%(1-threshold)

	#	Record the removed features 
	output_str += '\nRemoved Columns + Missing Percentage'

	if len(removed_columns) > 0:
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
	else:
		output_str += '\n\tNo columns were removed.'
		

	write_to(output_str, pth + 'column_decimation_info.txt')
	if newDataFramePath != '': df_decimated.to_csv(path_or_buf=newDataFramePath, index=False)

	return wuml.ensure_data_type(df_decimated, type_name=type(data).__name__)

def decimate_data_with_missing_entries(data, column_threshold=0.6, row_threshold=0.6,newDataFramePath=''):
	'''
		It will automatically remove rows and columns of a dataFrame with missing entries.
	'''

	dfo = wuml.ensure_DataFrame(data)
	dfSize = 'Data size:' + str(dfo.shape)

	mdp = np.array(identify_missing_data_per_feature(dfo))
	x = np.arange(1, len(mdp)+1)

	lp = lines(figsize=(10,5))
	lp.plot_line(x, mdp, 'Before Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=dfSize, subplot=121, ylim=[0,1], xTextLoc=0, yTextLoc=0.9,
					xtick_locations=x, xtick_labels=dfo.columns.to_numpy(), xticker_rotate=90)

	df = remove_columns_with_too_much_missing_entries(dfo, threshold=column_threshold)
	df_decimated = remove_rows_with_too_much_missing_entries(df, threshold=row_threshold, newDataFramePath=newDataFramePath)
	dfSize = 'Data size:' + str(df_decimated.shape)

	mdp = np.array(identify_missing_data_per_feature(df_decimated))
	x = np.arange(1, len(mdp)+1)

	lp.plot_line(x, mdp, 'After Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=dfSize, subplot=122, ylim=[0,1], xTextLoc=0, yTextLoc=0.9,
					xtick_locations=x, xtick_labels=df.columns.to_numpy(), xticker_rotate=90)
	lp.show()

	return df_decimated


def center_and_scale(wuData):
	X = wuData.get_data_as('ndarray')
	X = preprocessing.scale(X)

	wuData.df = pd.DataFrame(data=X, columns=wuData.df.columns)
	wuData.X = wuData.df.values
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

def split_training_test(data, label=None, data_name=None, data_path=None, save_as='ndarray', test_percentage=0.1, xdata_type="%.4f", ydata_type="%d"):
	
	X = wuml.ensure_numpy(data)
	if type(data).__name__ == 'wData': Y = data.Y
	else: Y = label
	if Y is None: raise ValueError('Error: The label Y is currently None, did you define it?')

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percentage, random_state=42)

	if data_path is not None:
		Train_dat = data_path + data_name + '_train.csv'
		Train_label_dat = data_path + data_name + '_train_label.csv'
	
		Test_dat = data_path + data_name + '_test.csv'
		Test_label_dat = data_path + data_name + '_test_label.csv'
	
	
		if save_as == 'ndarray':
			np.savetxt(Train_dat, X_train, delimiter=',', fmt=xdata_type) 
			np.savetxt(Train_label_dat, y_train, delimiter=',', fmt=ydata_type) 
	
			np.savetxt(Test_dat, X_test, delimiter=',', fmt=xdata_type) 
			np.savetxt(Test_label_dat, y_test, delimiter=',', fmt=ydata_type) 
		elif save_as == 'DataFrame':
			XTrain_df = pd.DataFrame(data=X_train, columns=data.df.columns)
			XTest_df =  pd.DataFrame(data=X_test, columns=data.df.columns)
	
			XTrain_df['label'] = y_train
			XTest_df['label'] = y_test
	
			XTrain_df.to_csv(Train_dat, index=False, header=True)
			XTest_df.to_csv(Test_dat, index=False, header=True)

	X_train = wuml.ensure_wData(X_train, column_names=data.df.columns)
	X_train.Y = y_train

	X_test = wuml.ensure_wData(X_test, column_names=data.df.columns)
	X_test.Y = y_test

	return [X_train, X_test, y_train, y_test]

def gen_10_fold_data(data_name, data_path='./data/'):

	xpath = data_path + data_name

	X = np.loadtxt(xpath + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt(xpath + '_label.csv', delimiter=',', dtype=np.int32)			

	fold_path = xpath + '/'
	if os.path.exists(fold_path): 
		pass
	else:
		os.mkdir(fold_path)

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


def use_cdf_to_map_data_between_0_and_1(data, output_type_name='wData'):
	X = ensure_numpy(data)
	n = X.shape[0]
	d = X.shape[1]

	newX = np.zeros(X.shape)

	for i in range(d):
		column = X[:,i]
		residual_dat = column[np.isnan(column) == False]
		minV = np.min(residual_dat) - 3

		if len(residual_dat) < 5:
			print('Error: column %d only has %d samples, you must at least have 6 samples for kde'%(i, len(residual_dat)))
			print('\nTry Removing this feature')
			sys.exit()

		if len(np.unique(residual_dat)) == 1:
			newX[:,i] = np.ones(n)
		else:
			Pₓ = wuml.KDE(residual_dat)
			for j, itm in enumerate(X[:,i]):
				if np.isnan(itm):
					newX[j,i] = np.nan
				else:
					newX[j,i] = Pₓ.integrate(minV, X[j,i])


	#	This ensures that the columns labels are copied correctly
	if type(data).__name__ == 'ndarray': 
		df = pd.DataFrame(newX)
	elif type(data).__name__ == 'wData': 
		df = pd.DataFrame(newX)
		df.columns = data.df.columns
	elif type(data).__name__ == 'DataFrame': 
		df = pd.DataFrame(newX)
		df.columns = data.columns
	elif type(data).__name__ == 'Tensor': 
		X = data.detach().cpu().numpy()
		df = pd.DataFrame(X)

	output = wuml.ensure_data_type(df, type_name=output_type_name)
	if output_type_name=='wData': 
		output.Y = data.Y
		output.label_column_name = data.label_column_name

	return output
