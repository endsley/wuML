#!/usr/bin/env python

import sys
import os
  
code_list = []
code_list.append('deal_with_missing_data.py')
code_list.append('get_N_samples_from_each_class.py')
code_list.append('map_data_to_between_0_and_1.py')
code_list.append('train_test_histogram.py')
code_list.append('use_cdf_to_map_data_to_between_0_and_1.py')
code_list.append('load_data_with_preprocess.py')
code_list.append('normalize_data_via_l1_l2.py')
code_list.append('ten_fold_cross_validation.py')
code_list.append('train_test_on_basic_network.py')


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
	

