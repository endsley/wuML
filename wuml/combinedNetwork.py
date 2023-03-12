
import sys
from wuml.basicNetwork import *
from wuml.type_check import *
import torch
import numpy as np


class combinedNetwork():
	def __init__(self, data, netStructureList, netInputDimList, costFunction, network_behavior_on_call, optimizer_steps_order=None, 
						early_exit_loss_threshold=0.0000001, 
						on_new_epoch_call_back = None, pickled_network_info=None, explainer=None,
						max_epoch=1000, data_mean=None, data_std=None,	force_network_to_use_CPU=False,
						X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, extra_dataType=None,
						network_usage_output_type='Tensor', learning_rate=0.001, lr_decay_rate=0.5, lr_decay_patience=50):
		'''
			data: must be a wData
			netStructureList: must be a list, can have multiple networks all working together
			on_new_epoch_call_back: end of each epoch calls this function
			lr_decay_rate: the current lr multiply by this value to decay, 1 would be no decay
			lr_decay_patience: the number of epochs which sees no improvement then trigger decay
			network_behavior_on_call:  what happens if you call  net(data) after instantiation
		'''
		if get_commandLine_input()[1] == 'disabled': max_epoch = 10
		if force_network_to_use_CPU: wuml.pytorch_device = 'cpu'
		self.explainer = explainer			# If this model's features were explained, we save the explainer here

		# the network is awared if the data was centered
		if wtype(data) is 'wData':
			self.μ = data.μ
			self.σ = data.σ
		else:
			self.μ = data_mean
			self.σ = data_std


		self.network_behavior_on_call = network_behavior_on_call
		self.on_new_epoch_call_back = on_new_epoch_call_back #set this as a callback at each function
		self.costFunction = costFunction
		self.device = wuml.get_current_device()

		if pickled_network_info is None:
			dat = wuml.ensure_wData(data)
	
			self.batch_size = data.batch_size
			self.netStructureList = netStructureList
			self.netInputDimList = netInputDimList
			self.optimizer_steps_order = optimizer_steps_order
			self.networkList = []
	
			self.X_dataType = X_dataType
			self.Y_dataType = Y_dataType
			self.extra_dataType=extra_dataType
			if wtype(extra_dataType) != 'list' and extra_dataType is not None:
				self.extra_dataType = [extra_dataType]
	
			# double check that extra data is properly defined
			if dat.extra_data_dictionary['extra_data'] is not None:
				if extra_dataType is None:
					ValueError('Error: If you have extra data within input, define the extra_dataType parameter as a list of data types.')
				if len(dat.extra_data_dictionary['extra_data']) != len(extra_dataType):
					raise ValueError('Error: If you have extra data within input, you must have a list of the same length for extra_dataType, e.g., extra_dataType=[torch.FloatTensor, torch.LongTensor]')
	
	
			self.trainLoader = dat.get_data_as('DataLoader')
			self.lr = learning_rate
			self.lr_decay_patience = lr_decay_patience
			self.lr_decay_rate = lr_decay_rate
			self.max_epoch = max_epoch
			self.early_exit_loss_threshold = early_exit_loss_threshold
	

			for NetStruct, dim in zip(netStructureList, netInputDimList):
				newNet = flexable_Model(dim, NetStruct)
				newNet.to(self.device)		# store the network weights in gpu or cpu device
				#newNet = basicNetwork(costFunction, dat, networkStructure=NetStruct, network_info_print=False, override_network_input_width_as=dim, max_epoch=self.max_epoch)
				self.networkList.append(newNet)

		else:
			self.lr = pickled_network_info['lr']
			self.max_epoch = pickled_network_info['max_epoch']
			self.netStructureList = pickled_network_info['netStructureList'] 
			self.networkList = pickled_network_info['networkList']
			self.extra_dataType = pickled_network_info['extra_dataType']
			self.batch_size = 32

			for model in self.networkList:
				model.to(self.device)
			

		self.info()

	def center_and_scaled(self, data):
		# the network is awared if the data was centered
		if self.μ is not None and self.σ is not None:
			X = ensure_numpy(data)
			X = (X - self.μ)/self.σ
			try:
				return ensure_data_type(X, type_name=wtype(data), ensure_column_format=True, column_names=data.columns)
			except:
				return ensure_data_type(X, type_name=wtype(data), ensure_column_format=True)
		else:
			return data

	def output_network_data_for_storage(self):
		net = {}
		net['name'] = self.__class__.__name__
		net['network_behavior_on_call'] = marshal.dumps(self.network_behavior_on_call.__code__)
		net['costFunction'] = marshal.dumps(self.costFunction.__code__)

		#if self.on_new_epoch_call_back is None: net['on_new_epoch_call_back'] = None
		#else: net['on_new_epoch_call_back'] = marshal.dumps(self.on_new_epoch_call_back.__code__)

		net['lr'] = self.lr
		net['max_epoch'] = self.max_epoch
		net['netStructureList'] = self.netStructureList
		net['networkList'] = self.networkList
		net['extra_dataType'] = self.extra_dataType
		net['μ'] = self.μ
		net['σ'] = self.σ
		if wtype(self.explainer) == 'explainer':
			net = self.explainer.output_network_data_for_storage(net)

		return net


	def info(self, printOut=True):
		info_str = 'All Networks\n' 
		info_str += '\tBatch size: %.d: \n'%self.batch_size
		info_str += '\tLearning rate: %.3f\n'%self.lr
		info_str += '\tMax number of epochs: %d\n'%self.max_epoch
		info_str += '\tCost Function: %s\n'%wuml.get_function_name(self.costFunction)
		if self.on_new_epoch_call_back is not None: info_str += '\tTrain Loop Callback: %s\n'%wuml.get_function_name(self.on_new_epoch_call_back)
		info_str += '\tDevice type: %r\n'%self.device

		for j, net in enumerate(self.networkList):
			info_str += '\tNetworks %d Structure\n'%(j) 

			for i in net.children():
				try: info_str += ('\t\t%s , %s\n'%(i,i.activation))
				except: info_str += ('\t\t%s \n'%(i))

		if printOut: wuml.jupyter_print(info_str)
		return info_str


	def fit(self, print_status=True):
		netParams = []
		netSchedulers = []
		netOptimizers = []

		for net in self.networkList:
			params = net.parameters()
			optimizer = torch.optim.Adam(params, lr=self.lr)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=self.lr_decay_rate, min_lr=1e-10, patience=50, verbose=False)	
					# factor is the decay rate
					# patience is how many epoch until trigger decay

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
				all_losses = self.costFunction(formatted_data, self.networkList)	

				if wtype(all_losses) == 'list': main_loss = all_losses[0]		# all_losses is a list of losses, but the 1st one is always the main loss
				else: main_loss = all_losses

				main_loss.backward()
				if self.optimizer_steps_order is None:
					for opt in netOptimizers: opt.step()
				else:
					self.optimizer_steps_order(netOptimizers)

				loss_history.append(main_loss.item())
	
			loss_avg = np.array(loss_history).mean()
			for sched in netSchedulers: sched.step(loss_avg)
			if loss_avg < self.early_exit_loss_threshold: break;


			#	The default status printing is just the main loss, but if you return a list of loss, 
			#	you can print your own style by adding the function on_new_epoch_call_back()
			if print_status:
				if self.on_new_epoch_call_back is not None:
					self.on_new_epoch_call_back(all_losses, (epoch+1), netSchedulers[0]._last_lr[0])
				else:
					if get_commandLine_input()[1] != 'disabled': 
						txt = '\tepoch: %d, Avg Loss/dimension: %.4f, Learning Rate: %.8f'%((epoch+1), loss_avg, enc_scheduler._last_lr[0])
						write_to_current_line(txt)
		
		for net in self.networkList: net.eval()

	def __call__(self, data, output_type='Tensor', out_structural=None):
		'''
			out_structural (mostly for classification purpose): None, '1d_labels', 'one_hot'
		'''

		data = ensure_wData(data)
		X = ensure_proper_model_input_format(data)
		X = ensure_tensor(X, dataType=torch.FloatTensor)
	
		if data.Y is None: formatted_data = [X,None]
		else:
			y = ensure_tensor(data.Y, dataType=torch.FloatTensor)
			formatted_data = [X,y]


		if wtype(data) == 'wData':
			extra_data = data.extra_data_dictionary['extra_data']
			if extra_data is not None:
				for j, xData in enumerate(extra_data): 
					newD = ensure_tensor(xData, dataType=self.extra_dataType[j])
					formatted_data.append(newD)

		all_net_output = self.network_behavior_on_call(formatted_data, self.networkList)	

		if wtype(all_net_output) == 'list':
			return wuml.cast_each_item_in_list_as(all_net_output, output_type)
		elif wtype(all_net_output) == 'Tensor':
			return ensure_data_type(all_net_output, type_name=output_type)
		else:
			return all_net_output
			





