
from wuml.basicNetwork import *
import torch
import numpy as np

class autoencoder():
	def __init__(self, bottleneck_size, X, Y=None, mask=None, costFunction=None,
						EncoderStructure=None, DecoderStructure=None, default_depth=4,
						default_activation_function='sigmoid',
						early_exit_loss_threshold=0.0000001, 
						on_new_epoch_call_back = None, max_epoch=1000, 	
						X_dataType=torch.FloatTensor, Y_dataType=torch.FloatTensor, 
						learning_rate=0.001):
		'''
			X : This should be wData type
			possible activation functions: softmax, relu, tanh, sigmoid, none
			simplify_network_for_storage: if a network is passed as this argument, we create a new network strip of unnecessary stuff
		'''
		bottleneck_size = int(bottleneck_size)
	
		if EncoderStructure is None or DecoderStructure is None:
			[self.encoder_structure, self.decoder_structure] = self.get_default_encoder_decoder(X, bottleneck_size, default_depth, default_activation_function)
		else:
			self.encoder_structure = EncoderStructure
			self.decoder_structure = DecoderStructure


			self.encoder = flexable_Model(X.shape[1], self.encoder_structure)
			self.decoder = flexable_Model(bottleneck_size, self.decoder_structure)


			self.trainLoader = X.get_data_as('DataLoader')

			self.lr = learning_rate
			self.max_epoch = max_epoch
			self.X_dataType = X_dataType
			self.Y_dataType = Y_dataType
			self.costFunction = costFunction
			self.NetStructure = networkStructure
			self.on_new_epoch_call_back = on_new_epoch_call_back #set this as a callback at each function
			self.model = flexable_Model(X.shape[1], networkStructure)
			self.network_output_in_CPU_during_usage = False



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
			print(i)
			if i == bottleneck_size: encoder_structure.append((int(i),'none'))
			else: encoder_structure.append((int(i),default_activation_function))

		for i in Rsizes:
			if i == d: decoder_structure.append((int(i),'none'))
			else: decoder_structure.append((int(i),default_activation_function))

		return [encoder_structure, decoder_structure]

