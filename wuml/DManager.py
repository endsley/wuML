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
	def __init__(self, X, Y=None, Y2=None): 
	# X: data, Y: label, Y2: 2nd label

		self.X = X
		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		self.X_Var = torch.tensor(self.X, requires_grad=False)

		self.Y = Y
		if Y is not None: self.Y_Var = torch.tensor(self.Y, requires_grad=False)

		self.Y2 = Y2
		if Y2 is not None: self.Y2_Var = torch.tensor(self.Y2, requires_grad=False)

	def __getitem__(self, index):

		if self.Y is None: items = [self.X[index], 0, index]
		else: items = [self.X[index], self.Y[index], index]

		if self.Y2 is not None:
			items.append(self.Y2[index])

		return items

	def __len__(self):
		try: return self.X.shape[0]
		except: return 0

