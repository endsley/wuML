
from wuml.basicNetwork import *
from wuml.type_check import *
import torch
import numpy as np

class combinedNetwork():
	def __init__(self, data, netStructureList, netInputDimList, costFunction,
						early_exit_loss_threshold=0.0000001, 
						on_new_epoch_call_back = None, max_epoch=1000, 	
						X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, extra_dataType=None,
						network_usage_output_type='Tensor', learning_rate=0.001):
		'''
			data: must be a wData
			netStructureList: must be a list, can have multiple networks all working together
			on_new_epoch_call_back: end of each epoch calls this function
		'''
		self.netStructureList = netStructureList
		self.netInputDimList = netInputDimList
		self.costFunction = costFunction
		self.networkList = []

		self.X_dataType = X_dataType
		self.Y_dataType = Y_dataType
		self.extra_dataType=extra_dataType
		if wtype(extra_dataType) != 'list':
			self.extra_dataType = [extra_dataType]




		self.trainLoader = data.get_data_as('DataLoader')
		self.lr = learning_rate
		self.max_epoch = max_epoch
		self.early_exit_loss_threshold = early_exit_loss_threshold

		self.on_new_epoch_call_back = on_new_epoch_call_back #set this as a callback at each function
		self.network_output_in_CPU_during_usage = False


		if torch.cuda.is_available(): self.device = 'cuda'
		else: self.device = 'cpu'
		for NetStruct, dim in zip(netStructureList, netInputDimList):
			#newNet = flexable_Model(dim, NetStruct)
			newNet = basicNetwork(costFunction, data, networkStructure=NetStruct, network_info_print=False, override_network_input_width_as=dim)
			#newNet.to(self.device)		# store the network weights in gpu or cpu device
			self.networkList.append(newNet)

		self.info()


	def info(self, printOut=True):
		info_str = 'All Networks\n' 
		for j, net in enumerate(self.networkList):
			single_info_str = 'Networks %d\n'%j 
			single_info_str += net.info(printOut=False)
			info_str += wuml.append_same_string_infront_of_block_of_string('\t', single_info_str)

		if printOut: wuml.jupyter_print(info_str)
		return info_str

	def fit(self, print_status=True):
		netParams = []
		netSchedulers = []
		netOptimizers = []

		for net in self.networkList:
			params = net.parameters()
			optimizer = torch.optim.Adam(params, lr=self.lr)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)

			netParams.append(params)
			netSchedulers.append(scheduler)
			netOptimizers.append(optimizer)

		for epoch in range(self.max_epoch):
	
			loss_history = []	
			for (i, data) in enumerate(self.trainLoader):
				x = Variable(data[0].type(self.X_dataType), requires_grad=False)
				y = Variable(data[1].type(self.Y_dataType), requires_grad=False)
				x= x.to(self.device, non_blocking=True )
				y= y.to(self.device, non_blocking=True )
				formatted_data = [x, y, data[2]]
			
				for j, dtype in enumerate(self.extra_dataType): 
					extraDat = Variable(data[j+3].type(dtype), requires_grad=False)
					extraDat= extraDat.to(self.device, non_blocking=True )
					formatted_data.append(extraDat)

				for opt in netOptimizers: opt.zero_grad()
				all_losses = self.costFunction(formatted_data, self.networkList)	# all_losses is a list of losses, but the 1st one is always the main loss
	
				all_losses[0].backward()
				for opt in netOptimizers: opt.step()
				loss_history.append(loss.item())
	
			loss_avg = np.array(loss_history).mean()
			for sched in netSchedulers: sched.step(loss_avg)
			if loss_avg < early_exit_loss_threshold: break;

			if print_status:
				if self.on_new_epoch_call_back is not None:
					self.on_new_epoch_call_back(all_losses, (epoch+1), enc_scheduler._last_lr[0])
				else:
					txt = '\tepoch: %d, Avg Loss/dimension: %.4f, Learning Rate: %.8f'%((epoch+1), loss_avg, enc_scheduler._last_lr[0])
					write_to_current_line(txt)



#	def objective_network(self, data, output_type='wData'):
#		x = ensure_tensor(data, dataType=torch.FloatTensor)
#		x̂ = self.encoder(x)
#		ẙ = self.midcoder(x̂)
#
#		ẙ = ensure_data_type(ẙ, type_name=output_type)
#		if wtype(ẙ) == 'wData': ẙ.Y = data.Y
#		return ẙ
#
#
#	def reduce_dimension(self, data, output_type='wData', outPath=None):
#		x = ensure_tensor(data, dataType=torch.FloatTensor)
#		x̂ = self.encoder(x)
#
#		x̂ = ensure_data_type(x̂, type_name=output_type)
#		#if output_type == 'ndarray' or self.network_usage_output_type == 'ndarray':
#		#	x̂ = x̂.detach().cpu().numpy()
#		#elif output_type == 'wData':
#		#	x̂ = ensure_wData(x̂)
#		#elif self.network_output_in_CPU_during_usage:
#		#	x̂ = x̂.detach().cpu()
#
#		if outPath is not None:
#			wD = ensure_wData(x̂)
#			wD.to_csv(outPath)
#
#		if wtype(x̂) == 'wData': x̂.Y = data.Y
#		return x̂
#
#	def __call__(self, data, output_type='Tensor', out_structural=None):
#		'''
#			out_structural (mostly for classification purpose): None, '1d_labels', 'one_hot'
#		'''
#
#		x = ensure_tensor(data, dataType=torch.FloatTensor)
#		x̂ = self.encoder(x)
#		ŷ = self.decoder(x̂)
#
#		if output_type == 'ndarray' or self.network_usage_output_type == 'ndarray':
#			return ŷ.detach().cpu().numpy()
#		elif self.network_output_in_CPU_during_usage:
#			return ŷ.detach().cpu()
#
#		return ŷ
#
#
#

