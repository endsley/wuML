

import numpy as np
import wuml
import torch
from torch.autograd import Variable



class l2x:
	def __init__(self, data, max_epoch=2000, learning_rate=0.001, data_imbalance_weights=None,
					X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, regularizer_weight=0,
					selector_network_structure=None, predictor_network_structure=None):
		self.data = data
		self.d = d = data.shape[1]
		self.max_epoch = max_epoch
		self.lr = learning_rate
		self.W = data_imbalance_weights
		self.X_dataType = X_dataType
		self.Y_dataType = Y_dataType
		self.regularizer_weight = regularizer_weight

		if selector_network_structure is None:
			self.θˢ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(d*2,'none')])
		else:
			self.θˢ = wuml.flexable_Model(d, selector_network_structure)


		if predictor_network_structure is None:
			self.θᴼ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(1,'none')])
		else:
			self.θᴼ = wuml.flexable_Model(d, predictor_network_structure)


		if torch.cuda.is_available(): 
			self.device = 'cuda'
			self.θˢ.to(self.device)		# store the network weights in gpu or cpu device
			self.θᴼ.to(self.device)		# store the network weights in gpu or cpu device
		else: self.device = 'cpu'


	def info(self, save_path=None):
		using_weights = True
		if self.W is None: using_weights = False

		txt = 'L2X Regression\n'
		txt += '\tDevice: %s\n'%(self.device)
		txt += '\tData Dimension: %d x %d\n'%(data.shape[0], data.shape[1])
		txt += '\tMax num of Epochs: %d\n'%(self.max_epoch)
		txt += '\tInitial Learning Rate: %.7f\n'%(self.lr)
		txt += '\tUsing weighted Samples: %r\n'%(using_weights)
		txt += '\tθˢ structure: %s\n'%(str(self.θˢ.networkStructure))
		txt += '\tθᴼ structure: %s\n'%(str(self.θᴼ.networkStructure))

		#save_path


	def get_Selector(self, x, round_result=False):
		[d, device] = [self.d, self.device]

		xᵃ = self.θˢ(x)
		xᵇ = xᵃ.view(-1, d, 2)	# reshape the data into num_of_samples x data dimension x num_of_groups 
		xᶜ = torch.nn.Softplus()(xᵇ)
		xᵈ = wuml.gumbel(xᶜ, device=device)		# Group matrix
		S = xᵈ[:,:,0]
		if round_result: S = torch.round(S)
		return S

	def __call__(self, x):
		x = wuml.ensure_tensor(x)

		S = self.get_Selector(x)
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

		regularizer = λ*torch.sum(S)
		loss = torch.sum(W*(y - ŷ)** 2)/n + regularizer
		return loss

	def train(self, print_status=True):
		trainLoader = self.data.get_data_as('DataLoader')
		param = list(self.θˢ.parameters()) + list(self.θᴼ.parameters())
		[ℓ, TL, mE, Dev] = [self.loss, trainLoader, self.max_epoch, self.device]
		[Xtype, Ytype] = [self.X_dataType, self.Y_dataType]

		wuml.run_SGD(ℓ, param, TL, Dev, lr=self.lr, max_epoch=mE, X_dataType=Xtype, Y_dataType=Ytype)


