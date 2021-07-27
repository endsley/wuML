#!/usr/bin/env python

import torch
import numpy as np
import sys
from sklearn import linear_model
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import collections

class flexable_Model(torch.nn.Module):
	def __init__(self, dataDim, networkStructure, learning_rate=0.001):
		super(flexable_Model, self).__init__()

		inDim = dataDim
		for l, Layer_info in enumerate(networkStructure):
			layer_width, activation_function = Layer_info
			φ = activation_function
			outDim = layer_width

			lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(inDim) + ', ' + str(outDim) + ' , bias=True)'
			exec(lr)
			exec('self.l' + str(l) + '.activation = "' + φ + '"')		#softmax, relu, tanh, sigmoid, none
			inDim = outDim

	def forward(self, x, y,ind):
		self.y0 = x
		for m, layer in enumerate(self.children(),0):
			var = 'self.y' + str(m+1)

			if layer.activation == 'none':
				cmd = 'self.yout = ' + var + ' = self.l' + str(m) + '(y' + str(m) + ')'
			else:
				cmd = 'self.yout = ' + var + ' = F.' + layer.activation + '(self.l' + str(m) + '(y' + str(m) + '))'
			exec(cmd)
		return self.yout

class basicNetwork:
	def __init__(self, costFunction, X, 
						Y=None, networkStructure=[(3,'relu'),(3,'relu'),(3,'none')], 
						on_new_epoch_call_back = None, max_epoch=1000, 	Torch_dataType=torch.FloatTensor, 
						learning_rate=0.001):
		'''
			possible activation functions: softmax, relu, tanh, sigmoid, none
		'''
		#	X should be in wuml format
		self.trainLoader = X.get_data_as('DataLoader')

		self.lr = learning_rate
		self.max_epoch = max_epoch
		self.Torch_dataType = Torch_dataType
		self.costFunction = costFunction
		self.NetStructure = networkStructure
		self.on_new_epoch_call_back = on_new_epoch_call_back #set this as a callback at each function
		self.model = flexable_Model(X.shape[1], networkStructure, learning_rate=0.001)

		self.info()

	def info(self):
		print('Network Info:')
		print('\tLearning rate: %.3f'%self.lr)
		print('\tNetwork Structure')
		for i in self.model.children():
			try:
				print('\t\t%s , %s'%(i,i.activation))
			except:
				print('\t\t%s '%(i))



	def on_new_epoch_call_back(self, loss_avg, num_of_epoch, lr):
		## Get Train, Test Accuracy, get loss, epoch, lr
		#db = self.db
		#db['debug'].output_current_network_state(loss_avg, num_of_epoch, lr, db['bigS'])
		#if loss_avg < 0.00001: return True	# early exit
		return False

	def train(self):
		model = self.model
	
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)	
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)
	
		for epoch in range(self.max_epoch):
	
			loss_list = []	
			for (i, data) in enumerate(self.trainLoader):
				[x, y, ind] = data
			
				x = Variable(x.type(self.Torch_dataType), requires_grad=False)
				y = Variable(y.type(self.Torch_dataType), requires_grad=False)
				optimizer.zero_grad()
				
				ŷ = model(x, y, ind)
				loss = self.costFunction(x, y, ŷ, ind)
				
				loss.backward()
				optimizer.step()
	
				loss_list.append(loss.item())

			loss_avg = np.array(loss_list).mean()
			scheduler.step(loss_avg)
			print(loss_avg)

		import pdb; pdb.set_trace()
			#if self.on_new_epoch_call_back is not None:
			#	early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
			#	if early_exit: break



