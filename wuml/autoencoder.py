
from wuml.basicNetwork import *
import torch
import numpy as np

#	x -> encoder -> x̂
#	x̂ -> encoder_linear_output -> ẙ	
#	x̂ -> decoder -> ŷ	
#	possible autoencoder objective λ could be 0
#	loss = (x - ŷ)ᒾ + λ * objective(ẙ, y)

class autoencoder():
	def __init__(self, bottleneck_size, X, Y=None, mask=None, costFunction=None,
						EncoderStructure=None, DecoderStructure=None, encoder_output_weight_structure=None, 
						default_depth=4, default_activation_function='relu',
						early_exit_loss_threshold=0.0000001, 
						on_new_epoch_call_back = None, max_epoch=1000, 	
						X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, network_usage_output_type='Tensor',
						learning_rate=0.001):
		'''
			X : This should be wData type
			possible activation functions: softmax, relu, tanh, sigmoid, none
			simplify_network_for_storage: if a network is passed as this argument, we create a new network strip of unnecessary stuff
		'''
		self.X = X
		bottleneck_size = int(bottleneck_size)
		self.encoder_output_weight_structure = encoder_output_weight_structure

		if EncoderStructure is None or DecoderStructure is None:
			[self.encoder_structure, self.decoder_structure] = self.get_default_encoder_decoder(X, bottleneck_size, default_depth, default_activation_function)
		else:
			self.encoder_structure = EncoderStructure
			self.decoder_structure = DecoderStructure


		self.encoder = flexable_Model(X.shape[1], self.encoder_structure)
		self.decoder = flexable_Model(bottleneck_size, self.decoder_structure)
		if encoder_output_weight_structure is not None:
			self.encoder_linear_output = flexable_Model(bottleneck_size, encoder_output_weight_structure)

		self.trainLoader = X.get_data_as('DataLoader')

		self.network_usage_output_type = network_usage_output_type
		self.lr = learning_rate
		self.max_epoch = max_epoch
		self.X_dataType = X_dataType
		self.Y_dataType = Y_dataType
		self.early_exit_loss_threshold = early_exit_loss_threshold
		if costFunction is None: costFunction = torch.nn.MSELoss()
		else: self.costFunction = costFunction
		self.on_new_epoch_call_back = on_new_epoch_call_back #set this as a callback at each function
		self.network_output_in_CPU_during_usage = False

		if torch.cuda.is_available(): 
			self.device = 'cuda'
			self.encoder.to(self.device)		# store the network weights in gpu or cpu device
			self.decoder.to(self.device)		# store the network weights in gpu or cpu device
			if encoder_output_weight_structure is not None:
				self.encoder_linear_output.to(self.device)

		else: self.device = 'cpu'

		self.info()


	def info(self, printOut=True):
		 
		info_str ='Autoencoder Info:\n'
		info_str += '\tLearning rate: %.3f\n'%self.lr
		info_str += '\tMax number of epochs: %d\n'%self.max_epoch
		info_str += '\tCost Function: %s\n'%str(self.costFunction)
		info_str += '\tTrain Loop Callback: %s\n'%str(self.on_new_epoch_call_back)
		info_str += '\tCuda Available: %r\n'%torch.cuda.is_available()
		info_str += '\tEncoder Structure\n'
		for i in self.encoder.children():
			try: info_str += ('\t\t%s , %s\n'%(i,i.activation))
			except: info_str += ('\t\t%s \n'%(i))

		if self.encoder_output_weight_structure is not None:
			info_str += '\tEncoder Extra Output weight Structure\n'
			for i in self.encoder_linear_output.children():
				try: info_str += ('\t\t%s , %s\n'%(i,i.activation))
				except: info_str += ('\t\t%s \n'%(i))


		info_str += '\tEncoder Structure\n'
		for i in self.decoder.children():
			try: info_str += ('\t\t%s , %s\n'%(i,i.activation))
			except: info_str += ('\t\t%s \n'%(i))

		if printOut: print(info_str)
		return info_str


	def get_default_encoder_decoder(self, X, bottleneck_size, default_depth, default_activation_function):
		# the default encoder and decoder depth are 3 layers
		d = X.X.shape[1]

		#step =  int(np.absolute(d - bottleneck_size)/default_depth)
		sizes = np.floor(np.linspace(d, bottleneck_size, default_depth))
		Rsizes = np.flip(sizes)

		sizes = np.delete(sizes, 0)	# remove the input dimension since it is already know
		Rsizes = np.delete(Rsizes, 0)

		encoder_structure = []
		decoder_structure = []

		for i in sizes:
			if i == bottleneck_size: encoder_structure.append((int(i),'none'))
			else: 
				encoder_structure.append((int(i),default_activation_function))
				#encoder_structure.append(('bn', True))

		for i in Rsizes:
			if i == d: decoder_structure.append((int(i),'none'))
			else: 
				decoder_structure.append((int(i),default_activation_function))
				#decoder_structure.append(('bn', True))

		return [encoder_structure, decoder_structure]


	def fit(self, print_status=True):
		enc_param = self.encoder.parameters()
		dec_param = self.decoder.parameters()

		[ℓ, TL, mE, Dev] = [self.costFunction, self.trainLoader, self.max_epoch, self.device]
		[Xtype, Ytype] = [self.X_dataType, self.Y_dataType]

		self.run_autoencoder_SGD(ℓ, enc_param, dec_param, TL, Dev, encoder=self.encoder, decoder=self.decoder, lr=self.lr, print_status=print_status,
				max_epoch=mE, X_dataType=Xtype, Y_dataType=Ytype, early_exit_loss_threshold=self.early_exit_loss_threshold,
				on_new_epoch_call_back = self.on_new_epoch_call_back)




	def run_autoencoder_SGD(self, loss_function, enc_param, dec_param, trainLoader, device, early_exit_loss_threshold=0.000000001,
					X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, 
					encoder=None, decoder=None, lr=0.001, print_status=True, max_epoch=1000,
					on_new_epoch_call_back=None):
	
	
		enc_optimizer = torch.optim.Adam(enc_param, lr=lr)	
		dec_optimizer = torch.optim.Adam(dec_param, lr=lr)	

		enc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( enc_optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)
		dec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( dec_optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)
	
		for epoch in range(max_epoch):
	
			loss_list = []	
			for (i, data) in enumerate(trainLoader):
				[x, y, ind] = data
	
				x = Variable(x.type(X_dataType), requires_grad=False)
				y = Variable(y.type(Y_dataType), requires_grad=False)
				x= x.to(device, non_blocking=True )
				y= y.to(device, non_blocking=True )
				enc_optimizer.zero_grad()
	
				x̂ = encoder(x)
				ŷ = decoder(x̂)

				if self.encoder_output_weight_structure is not None:
					ẙ = self.encoder_linear_output(x̂)
					ẙ = torch.squeeze(ẙ)
				else: ẙ = 0

				ŷ = torch.squeeze(ŷ)
				y = torch.squeeze(y)

				loss = loss_function(x, x̂, ẙ, y, ŷ, ind)
				if torch.isnan(loss): import pdb; pdb.set_trace()
	
				loss.backward()
				enc_optimizer.step()
				dec_optimizer.step()
	
				loss_list.append(loss.item())
	
			loss_avg = np.array(loss_list).mean()
			enc_scheduler.step(loss_avg)
			dec_scheduler.step(loss_avg)	

			
			if loss_avg < early_exit_loss_threshold: break;
			if print_status:
				txt = '\tepoch: %d, Avg Loss per dimension: %.4f, Learning Rate: %.8f'%((epoch+1), loss_avg/(self.X.batch_size*self.X.shape[1]), enc_scheduler._last_lr[0])
				write_to_current_line(txt)
	
			if on_new_epoch_call_back is not None:
				on_new_epoch_call_back(loss_avg, (epoch+1), enc_scheduler._last_lr[0])
	
			#if model is not None:
			#	if "on_new_epoch" in dir(model):
			#		early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
			#		if early_exit: break


	def reduce_dimension(self, data, output_type='Tensor', outPath=None):
		x = wuml.ensure_tensor(data, dataType=torch.FloatTensor)
		x̂ = self.encoder(x)

		if output_type == 'ndarray' or self.network_usage_output_type == 'ndarray':
			x̂ = x̂.detach().cpu().numpy()
		elif output_type == 'wData':
			x̂ = wuml.ensure_wData(x̂)
		elif self.network_output_in_CPU_during_usage:
			x̂ = x̂.detach().cpu()

		if outPath is not None:
			wD = wuml.ensure_wData(x̂)
			wD.to_csv(outPath)

		x̂.Y = data.Y
		return x̂

	def __call__(self, data, output_type='Tensor', out_structural=None):
		'''
			out_structural (mostly for classification purpose): None, '1d_labels', 'one_hot'
		'''

		x = wuml.ensure_tensor(data, dataType=torch.FloatTensor)

		#if type(data).__name__ == 'ndarray': 
		#	x = torch.from_numpy(data)
		#	x = Variable(x.type(self.X_dataType), requires_grad=False)
		#	x= x.to(self.device, non_blocking=True )
		#elif type(data).__name__ == 'Tensor': 
		#	x = Variable(x.type(self.X_dataType), requires_grad=False)
		#	x= x.to(self.device, non_blocking=True )
		#elif type(data).__name__ == 'wData': 
		#	x = data.get_data_as('Tensor')
		#else:
		#	raise

		x̂ = self.encoder(x)
		ŷ = self.decoder(x̂)

		if output_type == 'ndarray' or self.network_usage_output_type == 'ndarray':
			return ŷ.detach().cpu().numpy()
		elif self.network_output_in_CPU_during_usage:
			return ŷ.detach().cpu()

		return ŷ


