# wuml
## Chieh's quick ML library
## Pip Installation
```sh
pip install wuml
```


## Decimate Data with Missing data
```python
#!/usr/bin/env python

import wuml 

data = wuml.load_csv('../data/chem.exposures.csv', row_id_with_label=0)
dfSmall = wuml.decimate_data_with_missing_entries(data, column_threshold=0.95, row_threshold=0.9,newDataFramePath='') 

#	column_threshold=0.95, this will keep features that are at least 95% full

```
![Image](https://github.com/endsley/wuML/blob/main/img/Regression.png?raw=true)





## Example of basic regression 
```python
#!/usr/bin/env python


#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')]
#		2. Define a cost function
#		3. Call train()


import wuml
import numpy as np
import torch
import wplotlib


data = wuml.load_csv(xpath='examples/data/regress.csv', ypath='examples/data/regress_label.csv', batch_size=20)

def costFunction(x, y, ŷ, ind):
	ŷ = torch.squeeze(ŷ)
	return torch.sum((y- ŷ) ** 2)	

bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001)
bNet.train()

#	Test out on test data
newX = np.expand_dims(np.arange(0,5,0.1),1)
Ŷ = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

#	plot the results out
splot = wplotlib.scatter()
splot.add_plot(data.X, data.Y, marker='o')

lp = wplotlib.lines()	
lp.add_plot(newX, Ŷ)

splot.show(title='Basic Network Regression', xlabel='x-axis', ylabel='y-axis')

```
![Image](https://github.com/endsley/wuML/blob/main/img/Regression.png?raw=true)


## Example of basic classification 
```python
#!/usr/bin/env python

import wuml
import numpy as np
import torch
import torch.nn as nn
import wplotlib


#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'softmax')]
#		2. Define a cost function
#		3. Call train()

data = wuml.load_csv(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', batch_size=20)

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

Acc = wuml.accuracy(Y, Ŷ)
#Acc= accuracy_score(data.Y, Ŷ.cpu().numpy())
print('Accuracy: %.3f'%Acc)

```

#### **Code Output**
```python
Network Info:
	Learning rate: 0.001
	Max number of epochs: 3000
	Cuda Available: True
	Network Structure
		Linear(in_features=13, out_features=100, bias=True) , relu
		Linear(in_features=100, out_features=100, bias=True) , relu
		Linear(in_features=100, out_features=3, bias=True) , none
	epoch: 3000, Avg Loss: 0.0442, Learning Rate: 0.00001563Accuracy: 0.989

```



