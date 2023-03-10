#!/usr/bin/env python

import sys
import os

  
code_list = []
code_list.append('basic_shap_lime_explainer.py')
code_list.append('basicNN_explainer.py')
code_list.append('autoencoder_explainer.py')
code_list.append('complexNet_explained.py')
code_list.append('load_network_with_explainer.py')

#print('Running data_stats folder')
#for code in code_list:
#	print('\tRunning ' + code)
#	os.system('./' + code + ' disabled')


print('Make sure html folder exists')
if not os.path.exists('../html'): os.mkdir('../html')

print('Generate ipynb files')
for code in code_list:
	print('\tGenerating ' + code)
	os.system('p2j -o ' + code + ' > /dev/null 2>&1')
	ipynb = code.replace('.py', '.ipynb')
	os.system('jupyter nbconvert --to notebook --execute ' + ipynb + ' > /dev/null 2>&1')

	ran_ipynb = ipynb.replace('.ipynb', '.nbconvert.ipynb')
	os.system('mv ' + ran_ipynb + ' ' + ipynb)
	os.system('jupyter nbconvert --execute --to html ' + ipynb + ' > /dev/null 2>&1')

	html_ipynb = ipynb.replace('.ipynb', '.html')
	os.system('mv ' + ipynb + ' ../ipynb/' + ipynb)
	os.system('mv ' + html_ipynb + ' ../html/' + html_ipynb)
	

