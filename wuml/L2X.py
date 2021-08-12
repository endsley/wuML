

import numpy as np
import wuml
import torch
from torch.autograd import Variable



class l2x:
	def __init__(self, data, max_epoch=2000, learning_rate=0.001, data_imbalance_weights=None,
					X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor):
		self.data = data
		self.d = d = data.shape[1]
		self.max_epoch = max_epoch
		self.lr = learning_rate
		self.W = data_imbalance_weights
		self.X_dataType = X_dataType
		self.Y_dataType = Y_dataType

		self.θˢ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(d*2,'none')])
		self.θᴼ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(1,'none')])

		if torch.cuda.is_available(): 
			self.device = 'cuda'
			self.θˢ.to(self.device)		# store the network weights in gpu or cpu device
			self.θᴼ.to(self.device)		# store the network weights in gpu or cpu device
		else: self.device = 'cpu'


	def get_Selector(self, x):
		[d, device] = [self.d, self.device]

		xᵃ = self.θˢ(x)
		xᵇ = xᵃ.view(-1, d, 2)	# reshape the data into num_of_samples x data dimension x num_of_groups 
		xᶜ = torch.nn.Softplus()(xᵇ)
		xᵈ = wuml.gumbel(xᶜ, device=device)		# Group matrix
		S = xᵈ[:,:,0]
		return S

	def loss(self, x, y, ind):		# this function is not working currently
		n = len(ind)
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

		regularizer = 1*torch.sum(S)
		loss = torch.sum(W*(y - ŷ)** 2)/n + regularizer
		return loss

	def train(self, print_status=True):
		trainLoader = self.data.get_data_as('DataLoader')
		param = list(self.θˢ.parameters()) + list(self.θᴼ.parameters())
		[ℓ, TL, mE, Dev] = [self.loss, trainLoader, self.max_epoch, self.device]
		[Xtype, Ytype] = [self.X_dataType, self.Y_dataType]

		wuml.run_SGD(ℓ, param, TL, Dev, lr=self.lr, max_epoch=mE, X_dataType=Xtype, Y_dataType=Ytype)




#		joint_model_parameters = list(self.θˢ.parameters()) + list(self.θᴼ.parameters())
#		optimizer = torch.optim.Adam(joint_model_parameters, lr=self.lr)	
#		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)
#	
#		for epoch in range(self.max_epoch):
#	
#			loss_list = []	
#			for (i, data) in enumerate(trainLoader):
#				[x, y, ind] = data
#
#				x = Variable(x.type(self.X_dataType), requires_grad=False)
#				y = Variable(y.type(self.Y_dataType), requires_grad=False)
#				x= x.to(self.device, non_blocking=True )
#				y= y.to(self.device, non_blocking=True )
#				optimizer.zero_grad()
#				
#				l = self.loss(x, y, ind)
#				
#				l.backward()
#				optimizer.step()
#	
#				loss_list.append(l.item())
#
#			loss_avg = np.array(loss_list).mean()
#			scheduler.step(loss_avg)
#			if print_status:
#				txt = '\tepoch: %d, Avg Loss: %.4f, Learning Rate: %.8f'%((epoch+1), loss_avg, scheduler._last_lr[0])
#				write_to_current_line(txt)
#
#			if self.on_new_epoch_call_back is not None:
#				early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
#				if early_exit: break


