
import numpy as np
import wuml
import torch
import pandas as pd
from torch.autograd import Variable


def wtype(data):
	return type(data).__name__

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

def ensure_data_type(data, type_name='ndarray'):

	if type_name=='ndarray':
		return ensure_numpy(data)
	elif type_name=='DataFrame':
		return ensure_DataFrame(data)
	elif type_name=='Tensor':
		return ensure_tensor(data)
	elif type_name=='wData':
		return ensure_wData(data)


def ensure_list(data):
	if wtype(data) == 'str': 
		return [data]
	elif wtype(data) == 'list': 
		return data
	elif wtype(data) == 'NoneType': 
		return data
	elif wtype(data) == 'Index': 
		return data.tolist()
	if wtype(data) == 'ndarray': 
		return data.tolist()

def ensure_wData(data, column_names=None):
	if wtype(data) == 'ndarray': 
		return wuml.wData(X_npArray=data, column_names=column_names)
	elif wtype(data) == 'Index': 
		ArrayType = np.array(data.tolist())
		return wuml.wData(X_npArray=ArrayType, column_names=column_names)
	elif wtype(data) == 'wData': 
		return data
	elif wtype(data) == 'DataFrame': 
		return wuml.wData(dataFrame=data, column_names=column_names)
	elif wtype(data) == 'Tensor': 
		X = data.detach().cpu().numpy()
		return wuml.wData(X_npArray=X, column_names=column_names)
	elif wtype(data) == 'dimension_reduction': 
		return wuml.wData(X_npArray=data.Ӽ)
	elif wtype(data) == 'Series': 
		return wuml.wData(X_npArray=data.to_numpy(), column_names=column_names)

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


def ensure_numpy(data, rounding=None):
	if wtype(data) == 'ndarray': 
		if len(data.shape) == 1:
			X = np.atleast_2d(data).T
		else:
			X = data
	elif wtype(data) == 'wData': 
		X = data.df.values
	elif wtype(data) == 'DataFrame': 
		X = data.values
	elif wtype(data) == 'Tensor': 
		X = data.detach().cpu().numpy()
	elif wtype(data) == 'Series': 
		X = data.values
	elif np.isscalar(data):
		X = np.array([[data]])
	elif wtype(data) == 'dimension_reduction': 
		X = data.Ӽ
	else:
		raise ValueError('Unknown dataType %s'%wtype(data))

	if rounding is not None: X = np.round(X, rounding)
	return X


def ensure_tensor(data, dataType=torch.FloatTensor):
	if torch.cuda.is_available(): 
		device = 'cuda'
	else: self.device = 'cpu'

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





