
import wuml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from wuml.type_check import *

def evaluation(test_loader, name=None, model_best=None, epoch=None):
	# EVALUATION
	if model_best is None:
		# load best performing model
		model_best = torch.load(name + '.model')

	model_best.eval()
	loss = 0.
	N = 0.
	for indx_batch, test_batch in enumerate(test_loader):
		if hasattr(model, 'dequantization'):
			if model.dequantization:
				test_batch = test_batch + (1. - torch.rand(test_batch.shape))/2.
		loss_t = model_best.forward(test_batch, reduction='sum')
		loss = loss + loss_t.item()
		N = N + test_batch.shape[0]
	loss = loss / N

	if epoch is None:
		print(f'FINAL LOSS: nll={loss}')
	else:
		print(f'Epoch: {epoch}, val nll={loss}')

	return loss



def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
	nll_val = []
	best_nll = 1000.
	patience = 0

	# Main loop
	for e in range(num_epochs):
		# TRAINING
		model.train()
		for indx_batch, batch in enumerate(training_loader):
			if hasattr(model, 'dequantization'):
				if model.dequantization:
					batch = batch + (1. - torch.rand(batch.shape))/2.
			loss = model.forward(batch)

			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()

		# Validation
		loss_val = evaluation(val_loader, model_best=model, epoch=e)
		nll_val.append(loss_val)  # save for plotting

		if e == 0:
			print('saved!')
			torch.save(model, name + '.model')
			best_nll = loss_val
		else:
			if loss_val < best_nll:
				print('saved!')
				torch.save(model, name + '.model')
				best_nll = loss_val
				patience = 0

				samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
			else:
				patience = patience + 1

		if patience > max_patience:
			break

	nll_val = np.asarray(nll_val)

	return nll_val




class RealNVP(nn.Module):
	def __init__(self, nets, nett, num_flows, prior, D=2, dequantization=True):
		super(RealNVP, self).__init__()
		
		print('RealNVP by JT.')
		
		self.dequantization = dequantization
		
		self.prior = prior
		self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
		self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
		self.num_flows = num_flows
		
		self.D = D

	def coupling(self, x, index, forward=True):
		# x: input, either images (for the first transformation) or outputs from the previous transformation
		# index: it determines the index of the transformation
		# forward: whether it is a pass from x to y (forward=True), or from y to x (forward=False)
		
		(xa, xb) = torch.chunk(x, 2, 1)
		
		s = self.s[index](xa)
		t = self.t[index](xa)
		
		if forward:
			#yb = f^{-1}(x)
			yb = (xb - t) * torch.exp(-s)
		else:
			#xb = f(y)
			yb = torch.exp(s) * xb + t
		
		return torch.cat((xa, yb), 1), s

	def permute(self, x):
		return x.flip(1)

	def f(self, x):
		log_det_J, z = x.new_zeros(x.shape[0]), x
		for i in range(self.num_flows):
			z, s = self.coupling(z, i, forward=True)
			z = self.permute(z)
			log_det_J = log_det_J - s.sum(dim=1)

		return z, log_det_J

	def f_inv(self, z):
		x = z
		for i in reversed(range(self.num_flows)):
			x = self.permute(x)
			x, _ = self.coupling(x, i, forward=False)

		return x

	def forward(self, x, reduction='avg'):
		z, log_det_J = self.f(x)
		if reduction == 'sum':
			return -(self.prior.log_prob(z) + log_det_J).sum()
		else:
			return -(self.prior.log_prob(z) + log_det_J).mean()

	def sample(self, batchSize):
		z = self.prior.sample((batchSize, self.D))
		z = z[:, 0, :]
		x = self.f_inv(z)
		return x.view(-1, self.D)




class flow:
	def __init__(self, data, flow_mothod='realNVP', network_width=512, 
					lr = 1e-3, max_epochs=1000, max_patience=30, num_flows=8, dequantization=False):
		'''
			max_patience: an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
		'''
		X = ensure_wData(data)
		[X_train, X_test, y_train, y_test] = wuml.split_training_test(X, test_percentage=0.2, save_as='none')
		X_train_DL = X_train.get_data_as('DataLoader')
		X_test_DL = X_test.get_data_as('DataLoader')

		d = X.shape[1]
		self.netW = M = network_width
		self.lr = lr
		self.max_epochs = max_epochs
		self.max_patience = max_patience


		# scale (s) network
		self.S = nets = lambda: nn.Sequential(nn.Linear(d // 2, M), nn.LeakyReLU(), 
										nn.Linear(M, M), nn.LeakyReLU(), 
										nn.Linear(M, d // 2), nn.Tanh())
		
		# translation (t) network
		self.T = nett = lambda: nn.Sequential(nn.Linear(d // 2, M), nn.LeakyReLU(), 
										nn.Linear(M, M), nn.LeakyReLU(), 
										nn.Linear(M, d // 2))

		# Prior (a.k.a. the base distribution): Gaussian
		self.prior = prior = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))

		# Init RealNVP
		self.model = model = RealNVP(nets, nett, num_flows, prior, D=d, dequantization=dequantization)


		# OPTIMIZER
		self.optimizer = optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

		# Training procedure
		nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=max_epochs, 
							model=model, optimizer=optimizer, training_loader=X_train_DL, val_loader=X_test_DL)




	def generate_samples(self, num_of_samples):
		pass

	#def integrate(self, x0, x1):	# This only works for 1D data
	#	[result, error] = wuml.integrate(self, x0, x1)
	#	return result

	def __call__(self, data, return_log_likelihood=False):
		X = ensure_numpy(data)


