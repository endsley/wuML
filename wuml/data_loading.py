
from sklearn import preprocessing
import wuml
import pandas as pd
import numpy as np
import torch

class load_csv:
	def __init__(self, xpath=None, ypath=None, dataFrame=None, X_npArray=None, row_id_with_label=None , 
					sample_id_included=False, torchDataType=torch.FloatTensor, batch_size=20):
		'''
			row_id_with_label :  None = no labels, 0 = top row is the label
			dataFrame: if dataFrame is set, it ignores the path and use the dataFrame directly as the data itself
		'''
		if dataFrame is not None:
			self.df = dataFrame
		elif X_npArray is not None:
			self.df = pd.DataFrame(X_npArray)
		else:
			self.df = pd.read_csv (xpath, header=row_id_with_label)

		if ypath is not None: 
			self.Y = np.loadtxt(ypath, delimiter=',', dtype=np.float32)			

		self.batch_size = batch_size
		self.shape = self.df.shape
		self.torchDataType = torchDataType				
		self.torchloader = None

	def get_data_as(self, data_type): #'DataFrame', 'read_csv', 'Tensor'
		if data_type == 'read_csv': return self
		if data_type == 'DataFrame': return self.df
		if data_type == 'ndarray': 
			#self.df.values[subset]
			return self.df.values
		if data_type == 'DataLoader' and self.torchloader is None: 
			self.DM = wuml.DManager(self.df.values, self.Y)
			self.torchloader = DataLoader(dataset=self.DM, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
			return self.torchloader

	def __getitem__(self, item):
		import pdb; pdb.set_trace()
		print('kkk')
		return self.df.values[item]

	def __str__(self): 
		print(self.df)
 
