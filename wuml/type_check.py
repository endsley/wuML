
import numpy as np
import wuml
import torch
import pandas as pd
from torch.autograd import Variable


def data_type(data):
	return type(data).__name__

def ensure_data_type(data, type_name='ndarray'):

	if type_name=='ndarray':
		return ensure_numpy(data)
	elif type_name=='DataFrame':
		return ensure_DataFrame(data)
	elif type_name=='Tensor':
		return ensure_tensor(data)
	elif type_name=='wData':
		return ensure_wData(data)

def ensure_wData(data, column_names=None):
	if type(data).__name__ == 'ndarray': 
		return wuml.wData(X_npArray=data, column_names=column_names)
	elif type(data).__name__ == 'wData': 
		return data
	elif type(data).__name__ == 'DataFrame': 
		return wuml.wData(dataFrame=data, column_names=column_names)
	elif type(data).__name__ == 'Tensor': 
		X = data.detach().cpu().numpy()
		return wuml.wData(X_npArray=X, column_names=column_names)


def ensure_DataFrame(data, columns=None, index=None):
	if type(data).__name__ == 'ndarray': 
		df = pd.DataFrame(data, columns=columns, index=index)
	elif type(data).__name__ == 'wData': 
		df = data.df
		df.columns = data.df.columns
		df.index = data.df.index
	elif type(data).__name__ == 'DataFrame': 
		df = data
		#df.columns = data.df.columns
		#df.index = data.df.index
	elif type(data).__name__ == 'Tensor': 
		X = data.detach().cpu().numpy()
		df = pd.DataFrame(X)
		df.columns = data.df.columns
		df.index = data.df.index

	#if columns is not None:
	#	df.columns = columns

	return df


def ensure_numpy(data, rounding=None):
	if type(data).__name__ == 'ndarray': 
		if len(data.shape) == 1:
			X = np.atleast_2d(data).T
		else:
			X = data
	elif type(data).__name__ == 'wData': 
		X = data.df.values
	elif type(data).__name__ == 'DataFrame': 
		X = data.values
	elif type(data).__name__ == 'Tensor': 
		X = data.detach().cpu().numpy()
	elif type(data).__name__ == 'Series': 
		X = data.values
	elif np.isscalar(data):
		X = np.array([[data]])
	else:
		raise ValueError('Unknown dataType %s'%type(data).__name__)

	if rounding is not None: X = np.round(X, rounding)
	return X


def ensure_tensor(data, dataType=torch.FloatTensor):
	if torch.cuda.is_available(): 
		device = 'cuda'
	else: self.device = 'cpu'

	if type(data).__name__ == 'ndarray': 
		x = torch.from_numpy(data)
		x = Variable(x.type(dataType), requires_grad=False)
		X = x.to(device, non_blocking=True )
	elif type(data).__name__ == 'wData': 
		X = data.get_data_as('Tensor')
	elif type(data).__name__ == 'DataFrame': 
		X = data.values
	elif np.isscalar(data):
		X = np.array([[data]])
	elif type(data).__name__ == 'Tensor': 
		x = Variable(data.type(dataType), requires_grad=False)
		X = x.to(device, non_blocking=True )
	else:
		raise ValueError('Unknown dataType %s'%type(data).__name__)

	return X





