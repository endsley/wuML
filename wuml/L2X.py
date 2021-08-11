

import numpy as np
from wuml.type_check import *


class l2x:
	def __init__(self, data, max_epoch=2000, learning_rate=0.001):
		self.data = data
		d = data.shape[1]
		self.max_epoch = max_epoch

		self.θˢ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(d*d,'none')])
		self.θᴼ = wuml.flexable_Model(d, [(d,'relu'),(100,'relu'),(100,'relu'),(1,'none')])

	def get_Selector(self, Z):
		db = self.db
		Ḡ = db['d']
		ň = db['d'] # number of potential 1s
		ř = self.repeat_factor

		#	This one repeat gumbel on a larger network output and reshape them and multiply by permutation matrix							# Change here
		network_output = self.θˢ(Z) 
		#network_output = self.θˢ.predict(Z)		# RFF network

		output_reshaped = network_output.view(-1, ň*ř, Ḡ)	# reshape the data into num_of_samples x data dimension x num_of_groups 
		output_positive = torch.nn.Softplus()(output_reshaped)

		Gˢ = gumbel(output_positive, device=db['device'])		# Group matrix
		return Gˢ

	def loss(self, x, y, ind):		# this function is not working currently

		S = self.get_Selector(x)	
		x̂ = x*S
		ŷ = self.θᴼ(x̂)

		regularizer = db['L2X_λ']*torch.sum(S)
    	loss = torch.sum(weight * (input - target) ** 2)

		return loss

	def train(self, print_status=True):
		joint_model_parameters = list(self.θˢ.parameters()) + list(self.θᴼ.parameters())
		optimizer = torch.optim.Adam(joint_model_parameters, lr=self.lr)	
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)
	
		for epoch in range(self.max_epoch):
	
			loss_list = []	
			for (i, data) in enumerate(self.trainLoader):
				[x, y, ind] = data

				x = Variable(x.type(self.X_dataType), requires_grad=False)
				y = Variable(y.type(self.Y_dataType), requires_grad=False)
				x= x.to(self.device, non_blocking=True )
				y= y.to(self.device, non_blocking=True )
				optimizer.zero_grad()
				
				l = self.loss(x, y, ind)
				
				l.backward()
				optimizer.step()
	
				loss_list.append(l.item())

			loss_avg = np.array(loss_list).mean()
			scheduler.step(loss_avg)
			if print_status:
				txt = '\tepoch: %d, Avg Loss: %.4f, Learning Rate: %.8f'%((epoch+1), loss_avg, scheduler._last_lr[0])
				write_to_current_line(txt)

			if self.on_new_epoch_call_back is not None:
				early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
				if early_exit: break


