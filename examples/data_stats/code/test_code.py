#!/usr/bin/env python

import sys
import os

code_list = []
code_list.append('feature_wise_correlation.py')
code_list.append('feature_wise_HSIC.py')
code_list.append('get_stats_on_data_with_missing_entries.py')
code_list.append('show_feature_histogram.py')
code_list.append('weight_sample_by_rarity.py')

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
	

