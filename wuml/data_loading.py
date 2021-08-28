
from sklearn import preprocessing
import sys
import wuml
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class wData:
	def __init__(self, xpath=None, ypath=None, label_column_name=None, dataFrame=None, 
					X_npArray=None, row_id_with_label=None, sample_id_included=False, 
					label_type=None, #it should be either 'continuous' or 'discrete'
					torchDataType=torch.FloatTensor, batch_size=20, columns_to_ignore=None):
		'''
			row_id_with_label :  None = no labels, 0 = top row is the label
			dataFrame: if dataFrame is set, it ignores the path and use the dataFrame directly as the data itself
			ypath: loads the data as the label
			label_column_name: if the label is loaded together with xpath, this separates label into Y
		'''
		if dataFrame is not None:
			self.df = dataFrame
		elif X_npArray is not None:
			self.df = pd.DataFrame(X_npArray)
		else:
			self.df = pd.read_csv (xpath, header=row_id_with_label, index_col=False)

		if ypath is not None: 
			if label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')
			self.Y = np.loadtxt(ypath, delimiter=',', dtype=np.float32)			
			if label_type == 'discrete': self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0

		if label_column_name is not None:
			if label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')
			self.Y = self.df[label_column_name].values
			if label_type == 'discrete': self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0
			self.delete_column(label_column_name)

		if columns_to_ignore is not None: self.delete_column(columns_to_ignore)

		self.X = self.df.values
		self.batch_size = batch_size
		self.shape = self.df.shape
		self.torchDataType = torchDataType				
		self.torchloader = None

		if torch.cuda.is_available(): self.device = 'cuda'
		else: self.device = 'cpu'

	def delete_column(self, column_name):
		if type(column_name) == type([]):
			for name in column_name:
				del self.df[name]

		elif type(column_name) == type(''):
			del self.df[column_name]

	def info(self):
		print(self.df.info())

	def get_data_as(self, data_type): #'DataFrame', 'read_csv', 'Tensor'
		if data_type == 'wData': return self
		if data_type == 'Tensor': 
			x = torch.from_numpy(self.df.values)
			x = Variable(x.type(self.torchDataType), requires_grad=False)
			X = x.to(self.device, non_blocking=True )
			return X
		if data_type == 'DataFrame': return self.df
		if data_type == 'ndarray': 
			#self.df.values[subset]
			return self.df.values
		if data_type == 'DataLoader' and self.torchloader is None: 
			self.DM = wuml.DManager(self.df.values, self.Y)
			self.torchloader = DataLoader(dataset=self.DM, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
			return self.torchloader

	def to_csv(self, path, add_row_indices=False, include_column_names=True):
		self.df.to_csv(path, index=add_row_indices, header=include_column_names)

	def __getitem__(self, item):
		return self.df.values[item]

	def __str__(self): 
		print(self.df)
 
