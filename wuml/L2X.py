

import numpy as np
import wuml
import torch
from torch.autograd import Variable
import torch.nn.functional as f

#torch.autograd.set_detect_anomaly(True)

class l2x:
	def __init__(self, data, max_epoch=2000, learning_rate=0.001, data_imbalance_weights=None,
					X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, regularizer_weight=0,
					selector_network_structure=None, predictor_network_structure=None, 
					pre_trained_file=None, use_binary=True):
		self.data = data
		self.d = d = data.shape[1]
		self.max_epoch = max_epoch
		self.lr = learning_rate
		self.W = data_imbalance_weights
		self.X_dataType = X_dataType
		self.Y_dataType = Y_dataType
		self.regularizer_weight = regularizer_weight
		self.use_binary = use_binary
		self.pre_trained_file = pre_trained_file

		if pre_trained_file is None:
			if selector_network_structure is None:
				#self.θˢ = wuml.flexable_Model(d, [(d,'relu'),(1000,'relu'),(1000,'relu'),(d*2,'none')])
				self.θˢ = wuml.flexable_Model(d, [(d,'relu'),(1000,'relu'),(1000,'relu'),(d*d,'none')])
			else:
				self.θˢ = wuml.flexable_Model(d, selector_network_structure)
	
			if predictor_network_structure is None:
				self.θᴼ = wuml.flexable_Model(d, [(d,'relu'),(1000,'relu'),(1000,'relu'),(1,'none')])
			else:
				self.θᴼ = wuml.flexable_Model(d, predictor_network_structure)
		else:
			θ = wuml.pickle_load(pre_trained_file)
			self.θˢ = θ['θˢ']
			self.θᴼ = θ['θᴼ']

		if torch.cuda.is_available(): 
			self.device = 'cuda'
			self.θˢ.to(self.device)		# store the network weights in gpu or cpu device
			self.θᴼ.to(self.device)		# store the network weights in gpu or cpu device
		else: self.device = 'cpu'
		self.info()

	def export_network(self, save_path):
		network = {}
		network['θᴼ'] = self.θᴼ
		network['θˢ'] = self.θˢ

		wuml.pickle_dump(network, save_path)


	def info(self, save_path=None):
		using_weights = True
		if self.W is None: using_weights = False

		txt = 'L2X Regression\n'
		txt += '\tDevice: %s\n'%(self.device)
		txt += '\tBatch size: %d\n'%(self.data.batch_size)
		txt += '\tData Dimension: %d x %d\n'%(self.data.shape[0], self.data.shape[1])
		txt += '\tUse Binary One Hot: %r\n'%(self.use_binary)
		txt += '\tUse Pre-train File: %s\n'%(str(self.pre_trained_file))
		txt += '\tMax num of Epochs: %d\n'%(self.max_epoch)
		txt += '\tInitial Learning Rate: %.7f\n'%(self.lr)
		txt += '\tUsing weighted Samples: %r\n'%(using_weights)
		txt += '\tθˢ structure: %s\n'%(str(self.θˢ.networkStructure))
		txt += '\tθᴼ structure: %s\n'%(str(self.θᴼ.networkStructure))
		print(txt)
		#save_path


	def get_Selector(self, x, round_result=False, test_phase=False):
		x = wuml.ensure_tensor(x)
		[d, device] = [self.d, self.device]

		if test_phase:
			if self.use_binary:
				xᵃ = self.θˢ(x)
				xᵇ = xᵃ.view(-1, d, 2)	# reshape the data into num_of_samples x data dimension x num_of_groups 
				xᶜ = torch.nn.Softplus()(xᵇ)

				S = torch.argmin(xᶜ, dim=2)
			else:
				xᵃ = self.θˢ(x)
				q = int(xᵃ.shape[1]/d)
				xᵇ = xᵃ.view(-1, q, d)	# reshape the data into num_of_samples x data dimension x num_of_groups 
				xᶜ = torch.nn.Softplus()(xᵇ)
				xᶜ = f.normalize(xᶜ, p=1, dim=2)

				[S,ind] = torch.max(xᶜ, dim=1)
		else:
			if self.use_binary:
				xᵃ = self.θˢ(x)
				xᵇ = xᵃ.view(-1, d, 2)	# reshape the data into num_of_samples x data dimension x num_of_groups 
				xᶜ = torch.nn.Softplus()(xᵇ)
				xᶜ = f.normalize(xᶜ, p=1, dim=2)
	
				xᵈ = wuml.gumbel(xᶜ, device=device)		# Group matrix
				S = xᵈ[:,:,0]
			else:
				xᵃ = self.θˢ(x)
				q = int(xᵃ.shape[1]/d)
				xᵇ = xᵃ.view(-1, q, d)	# reshape the data into num_of_samples x data dimension x num_of_groups 
	
				xᶜ = torch.nn.Softplus()(xᵇ)
				xᶜ = f.normalize(xᶜ, p=1, dim=2)
				xᵈ = wuml.gumbel(xᶜ, device=device)		# Group matrix
				
				[S,ind] = torch.max(xᵈ, dim=1)





		if torch.isnan(S[0,0]): 
			print('\n\nNan detected within get_Selector function in L2X')
			import pdb; pdb.set_trace() 

		if round_result: S = torch.round(S)
		return S

	def __call__(self, x):
		x = wuml.ensure_tensor(x)

		S = self.get_Selector(x, test_phase=True)
		x̂ = x*S
		ŷ = self.θᴼ(x̂)
		return ŷ

	def loss(self, x, y, ind):		# this function is not working currently
		[n, λ] = [len(ind), self.regularizer_weight]

		if self.W is None:
			W = torch.ones(n, 1)
			W = Variable(W.type(self.X_dataType), requires_grad=False)
		else:
			W = self.W[ind]
			W = torch.from_numpy(W)
			W = Variable(W.type(self.X_dataType), requires_grad=False)

		W = W.to(self.device, non_blocking=True )
		W = torch.squeeze(W)

		S = self.get_Selector(x)
		x̂ = x*S
		ŷ = self.θᴼ(x̂)
		ŷ = torch.squeeze(ŷ)

		if λ == 0:
			regularizer = λ*torch.sum(S)
			loss = torch.sum(W*((y - ŷ)** 2))/n + regularizer
		else:
			loss = torch.sum(W*((y - ŷ)** 2))/n

		#if loss < 0.01:
		#	import pdb; pdb.set_trace()

		if torch.isnan(loss): import pdb; pdb.set_trace()
		return loss

	def on_new_epoch_call_back(self, loss_avg, epoch, lr):
		pass

		#if epoch%2 == 0:
		#	for param in self.θˢ.parameters(): 
		#		print(param[0:10,0:10], '\n')
		#		break

			#for param in self.θᴼ.parameters(): 
			#	print(param[0:10,0:10])
			#	break

		#	S = self.get_Selector(self.data.X, round_result=True)
		#	import pdb; pdb.set_trace()

	def train(self, print_status=True):
		trainLoader = self.data.get_data_as('DataLoader')
		param = list(self.θˢ.parameters()) + list(self.θᴼ.parameters())
		[ℓ, TL, mE, Dev] = [self.loss, trainLoader, self.max_epoch, self.device]
		[Xtype, Ytype] = [self.X_dataType, self.Y_dataType]

		wuml.run_SGD(ℓ, param, TL, Dev, lr=self.lr, 
					max_epoch=mE, X_dataType=Xtype, Y_dataType=Ytype,
					on_new_epoch_call_back=self.on_new_epoch_call_back)


