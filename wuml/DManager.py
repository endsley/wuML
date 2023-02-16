#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
#from src.tools.path_tools import *
#from distances import *
#from src.tools.terminal_print import *
import numpy as np

class DManager(Dataset):
	def __init__(self, X, Y=None, extra_data_dictionary=None): 
	# X: data, Y: label, extra_data_dictionary: dictionary of extra data for anything

		self.X = X
		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		self.X_Var = torch.tensor(self.X, requires_grad=False)

		self.Y = Y
		if Y is not None: self.Y_Var = torch.tensor(self.Y, requires_grad=False)
		self.extra_data_dictionary = extra_data_dictionary


	def __getitem__(self, index):

		if self.extra_data_dictionary is None:
			if self.Y is None: items = [self.X[index], 0, index]
			else: items = [self.X[index], self.Y[index], index]
		else:
			if self.Y is None: Yout = 0
			else: Yout = self.Y[index]
			items = [self.X[index], Yout, index]
			for dat in self.extra_data_dictionary['numpy']:
				items.append(dat[index])

		return items

	def __len__(self):
		try: return self.X.shape[0]
		except: return 0

