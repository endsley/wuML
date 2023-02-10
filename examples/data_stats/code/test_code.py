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

