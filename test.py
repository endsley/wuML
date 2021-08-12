#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
	
wuml.set_terminal_print_options(precision=2)
data = wuml.wData('examples/data/Chem_decimated_imputed.csv', 
					label_type='continuous', row_id_with_label=0, 
					label_column_name='finalga_best', columns_to_ignore=['id'])

imbW = wuml.wData('examples/data/Chem_sample_weights.csv')
P = wuml.l2x(data, max_epoch=2000, learning_rate=0.001, data_imbalance_weights=imbW)
P.train()
import pdb; pdb.set_trace()






#import wuml
#import numpy as np
#import torch
#import wplotlib
#
#
#data = wuml.wData(xpath='examples/data/regress.csv', ypath='examples/data/regress_label.csv', batch_size=20, label_type='continuous')
#
#def costFunction(x, y, ŷ, ind):
#	ŷ = torch.squeeze(ŷ)
#	return torch.sum((y- ŷ) ** 2)	
#
#bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001)
#bNet.train()
#
##	Test out on test data
#newX = np.expand_dims(np.arange(0,5,0.1),1)
#Ŷ = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
#
##	plot the results out
#splot = wplotlib.scatter()
#splot.add_plot(data.X, data.Y, marker='o')
#
#lp = wplotlib.lines()	
#lp.add_plot(newX, Ŷ)
#
#splot.show(title='Basic Network Regression', xlabel='x-axis', ylabel='y-axis')









import wuml
import numpy as np
import torch
import torch.nn as nn
import wplotlib
from sklearn.metrics import accuracy_score

data = wuml.wData(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', batch_size=20, label_type='discrete')

def costFunction(x, y, ŷ, ind):
	lossFun = nn.CrossEntropyLoss() 
	loss = lossFun(ŷ, y) #weird from pytorch, dim of y is 1, and ŷ is 20x3	
	return loss


#It is important for pytorch that with classification, you need to define Y_dataType=torch.int64
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(3,'none')], 
						Y_dataType=torch.int64, max_epoch=3000, learning_rate=0.001)
bNet.train()
netOutput = bNet(data.X)

#	Output Accuracy
_, Ŷ = torch.max(netOutput, 1)
Acc= accuracy_score(data.Y, Ŷ.cpu().numpy())
print('Accuracy: %.3f'%Acc)

