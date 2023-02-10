#!/usr/bin/env python

import sys
import os

code_list = []
code_list.append('classify.py')
code_list.append('tenfold_bagging_classifier.py')

print('Running data_stats folder')
for code in code_list:
	print('\tRunning ' + code)
	os.system('./' + code + ' disabled')

print('Generate ipynb files')
for code in code_list:
	print('\tGenerating ' + code)
	os.system('p2j -o ' + code)
	ipynb = code.replace('.py', '.ipynb')
	os.system('jupyter nbconvert --to notebook --execute ' + ipynb + ' > /dev/null 2>&1')

	ran_ipynb = ipynb.replace('.ipynb', '.nbconvert.ipynb')
	os.system('mv ' + ran_ipynb + ' ' + ipynb)
	os.system('mv ' + ipynb + ' ../ipynb/' + ipynb)
	

