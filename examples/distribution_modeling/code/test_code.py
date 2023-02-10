#!/usr/bin/env python

import sys
import os
  
code_list = []
code_list.append('basic_exponential_MLE_modeling.py')
code_list.append('basicKDE_estimate.py')
code_list.append('flow_example.py')
code_list.append('flow_prob.py')
code_list.append('model_exponential_distr_via_MLE_with_cdf.py')

print('Running data_stats folder')
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
	

