#!/usr/bin/env python

import sys
import os
  
code_list = []
code_list.append('autoencoder.py')
code_list.append('autoencoder_regression.py')
code_list.append('basicClassification.py')
code_list.append('basicRegression.py')
code_list.append('load_use_network.py')
code_list.append('gumble.py')
code_list.append('HSIC_as_objective.py')
code_list.append('weighted_regression.py')
code_list.append('complexNet.py')


print('Running ' + os.getcwd() + ' folder')
for code in code_list:
	print('\tRunning ' + code)
	os.system('./' + code + ' disabled')

print('Generate ipynb files')
for code in code_list:
	print('\tGenerating ' + code)
	os.system('p2j -o ' + code + ' > /dev/null 2>&1')
	ipynb = code.replace('.py', '.ipynb')
	os.system('jupyter nbconvert --to notebook --execute ' + ipynb + ' > /dev/null 2>&1')

	ran_ipynb = ipynb.replace('.ipynb', '.nbconvert.ipynb')
	os.system('mv ' + ran_ipynb + ' ' + ipynb)
	os.system('mv ' + ipynb + ' ../ipynb/' + ipynb)
	

