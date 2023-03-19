
import numpy as np
import wuml
import torch
import pandas as pd
from torch.autograd import Variable


def wtype(data):
	return type(data).__name__

def cast_each_item_in_list_as(list_of_data, cast_type):
	casted_list = [] 
	for dat in list_of_data:
		casted_dat = ensure_data_type(dat, type_name=cast_type)
		casted_list.append(casted_dat)

	return casted_list

def get_function_name(function):
	if callable(function):
		return function.__name__
	else:
		return 'None'

def type_check_with_error(data, desired_type, function_name=''):
	if wtype(data) != desired_type:
		raise ValueError('For function %s, you must input type as %s'%(function_name, desired_type))


def ensure_column(data):
	X = ensure_numpy(data)
	if len(X.shape) == 1:
		X = X.expand_dim().T

	elif len(X.shape) == 2:
		if X.shape[0] == 1:
			X = X.T
	return X

def is_binary_label(data):
	y = ensure_numpy(data)
	y_labels = np.unique(y)

	if len(y_labels) == 2 and (1 in y_labels) and (0 in y_labels):
		return True
	else:
		return False

def ensure_label(data, y=None, y_column_name=None):
	NP = ensure_numpy
	S = np.squeeze

	if y is not None:
		Y = S(NP(y))
	elif y_column_name is not None:
		Y = data[y_column_name].values
	elif wtype(data) == 'wData':
		Y = data.Y
	else: raise ValueError('Undefined label Y')

	return Y

def ensure_data_type(data, type_name='ndarray', ensure_column_format=True, column_names=None):

	if type_name=='ndarray':
		return ensure_numpy(data, ensure_column_format=ensure_column_format)
	elif type_name=='DataFrame':
		return ensure_DataFrame(data, columns=column_names)
	elif type_name=='Tensor':
		return ensure_tensor(data)
	elif type_name=='wData':
		return ensure_wData(data, column_names=column_names)


def ensure_list(data):
	if wtype(data) == 'str': 
		return [data]
	elif wtype(data) == 'list': 
		return data
	elif wtype(data) == 'NoneType': 
		return data
	elif wtype(data) == 'Int64Index': 
		return data.values.tolist()
	elif wtype(data) == 'Index': 	
		return data.tolist()
	elif wtype(data) == 'ndarray': 
		return data.tolist()
	elif wtype(data) == 'wData': 
		npD = data.df.to_numpy()
		npD = np.squeeze(npD)
		return npD.tolist()
	elif type_name=='DataFrame':
		npD = data.to_numpy()
		npD = np.squeeze(npD)
		return npD.tolist()


def ensure_wData(data, column_names=None, extra_data=None):
	if wtype(data) == 'ndarray': 
		return wuml.wData(X_npArray=data, column_names=column_names, extra_data=extra_data)
	elif wtype(data) == 'Index': 
		ArrayType = np.array(data.tolist())
		return wuml.wData(X_npArray=ArrayType, column_names=column_names, extra_data=extra_data)
	elif wtype(data) == 'wData': 
		return data
	elif wtype(data) == 'DataFrame': 
		return wuml.wData(dataFrame=data, column_names=column_names, extra_data=extra_data)
	elif wtype(data) == 'Tensor': 
		X = data.detach().cpu().numpy()
		return wuml.wData(X_npArray=X, column_names=column_names, extra_data=extra_data)
	elif wtype(data) == 'dimension_reduction': 
		return wuml.wData(X_npArray=data.Ӽ, extra_data=extra_data)
	elif wtype(data) == 'Series': 
		return wuml.wData(X_npArray=data.to_numpy(), column_names=column_names, extra_data=extra_data)
	elif wtype(data) == 'float64': 
		return wuml.wData(X_npArray=np.array([data]), column_names=column_names, extra_data=extra_data)


def ensure_DataFrame(data, columns=None, index=None):
	columns = ensure_list(columns)

	if wtype(data) == 'ndarray': 
		df = pd.DataFrame(data, columns=columns, index=index)
	elif wtype(data) == 'wData': 
		df = data.df
		df.columns = data.df.columns
		df.index = data.df.index
	elif wtype(data) == 'DataFrame': 
		df = data
		if columns is not None: df.columns = columns
		#df.index = data.df.index
	elif wtype(data) == 'Tensor': 
		X = data.detach().cpu().numpy()
		df = pd.DataFrame(X)
		df.columns = data.df.columns
		df.index = data.df.index
	elif wtype(data) == 'dimension_reduction': 
		df = pd.DataFrame(data.Ӽ)

	return df


def ensure_numpy(data, rounding=None, ensure_column_format=True):

	if wtype(data) == 'ndarray': 
		if ensure_column_format and len(data.shape) == 1:
			X = np.atleast_2d(data).T
		else:
			X = data
	elif wtype(data) == 'Index': 
		return data.to_numpy()
	elif wtype(data) == 'list': 
		return np.array(data)
	elif wtype(data) == 'wData': 
		X = data.df.values
	elif wtype(data) == 'DataFrame': 
		X = data.values
	elif wtype(data) == 'Tensor': 
		X = data.detach().cpu().numpy()
	elif wtype(data) == 'Series': 
		X = data.values
	elif wtype(data) == 'Int64Index': 
		return data.to_numpy()
	elif np.isscalar(data):
		X = np.array([[data]])
	elif wtype(data) == 'dimension_reduction': 
		X = data.Ӽ
	else:
		raise ValueError('Unknown dataType %s'%wtype(data))

	if rounding is not None: X = np.round(X, rounding)
	return X


def ensure_tensor(data, dataType=torch.FloatTensor):
	device = wuml.get_current_device()

	if wtype(data) == 'ndarray': 
		x = torch.from_numpy(data)
		x = Variable(x.type(dataType), requires_grad=False)
		X = x.to(device, non_blocking=True )
	elif wtype(data) == 'wData': 
		X = data.get_data_as('Tensor')
	elif wtype(data) == 'DataFrame': 
		X = data.values
	elif np.isscalar(data):
		X = np.array([[data]])
	elif wtype(data) == 'Tensor': 
		x = Variable(data.type(dataType), requires_grad=False)
		X = x.to(device, non_blocking=True )
	elif wtype(data) == 'dimension_reduction': 
		x = torch.from_numpy(data.Ӽ)
		x = Variable(x.type(dataType), requires_grad=False)
		X = x.to(device, non_blocking=True )

	else:
		raise ValueError('Unknown dataType %s'%wtype(data))

	return X


def ensure_proper_model_input_format(data):
	X = ensure_numpy(data)	
	if len(X.shape) == 2:
		if X.shape[1] == 1:
			X = X.T
	if len(X.shape) == 1:
		X = np.expand_dims(X, axis=0)

	return X





if __name__ == "__main__":

	y = [1, 0, 0, 0, 1]
	print(is_binary_label(y))

	y = [True, False, False, True, True]
	print(is_binary_label(y))

	y = ['red', 'red', 'red', 'blue', 'blue']
	print(is_binary_label(y))
