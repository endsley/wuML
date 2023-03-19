#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

from wplotlib import histograms
import numpy as np
import wuml
import torch
from wuml.type_check import *


#str.ljust(s, width[, fillchar])
#str.rjust(s, width[, fillchar])
#str.center(s, width[, fillchar])


def pretty_np_array(m, front_tab='', verticalize=False, title=None, auto_print=False, end_space='', round_value=3):

	M = ensure_wData(m)
	M.round(rounding=round_value)

	#if wtype(m) == 'float64': 
	#	import pdb; pdb.set_trace()

	if auto_print: wuml.jupyter_print(M)
	else: 
		out_str = str(M)
		L1 = out_str.split('\n')
		L1_max_width = len(max(L1, key=len))
		if wtype(title) == 'str':
			t1 = str.center(title, L1_max_width)
			out_str = t1 + '\n' + out_str + end_space
		else:
			out_str = out_str + end_space

		return out_str


#	This portion is deprecated, we now use dataframe instead of np array.

#	cName = None
#	if wtype(m) =='DataFrame': 
#		out_str = str(m.round(round_value))
#	else:
#		m = ensure_wData(m)
#		cName = ensure_numpy(m.columns)
#		m = wuml.ensure_numpy(m)
#
#		try: m = np.round(m, round_value)
#		except: pass
#		if cName is not None: m = np.vstack((cName,m))
#		m = str(m)
#	
#		if verticalize:
#			if len(m.shape) == 1:
#				m = np.atleast_2d(m).T
#
#		out_str = front_tab + str(m).replace('\n ','\n' + front_tab).replace('[[','[').replace(']]',']') + end_space + '\n'
#		out_str = str(out_str).replace('.]',']')
#
#		if wtype(title) == 'str':
#			L1 = out_str.split('\n')
#			L1_max_width = len(max(L1, key=len))
#			t1 = str.center(title, L1_max_width)
#			out_str = t1 + '\n' + out_str	
#	if auto_print: wuml.jupyter_print(out_str)
#	else: return out_str



def append_same_string_infront_of_block_of_string(front_string, back_block_string):

	new_str = front_string.join(back_block_string.splitlines(True))
	new_str = front_string + new_str
	return new_str


def block_two_string_concatenate(str1, str2, spacing='\t', add_titles=[], auto_print=False):
	str1 = str(str1)
	str2 = str(str2)

	L1 = str1.split('\n')
	#L2 = str2.strip().split('\n')
	L2 = str2.split('\n')

	if len(L1) > len(L2):
		Δ = len(L1) - len(L2)
		for ι in range(Δ):
			L2.append('\n')

	if len(add_titles) == 2:
		L1_max_width = len(max(L1, key=len))
		L2_max_width = len(max(L2, key=len))
		t1 = str.center(add_titles[0], L1_max_width)
		t2 = str.center(add_titles[1], L2_max_width)
		L1.insert(0,t1)
		L2.insert(0,t2)

	max_width = len(max(L1, key=len))
	outS = ''
	for l1, l2 in zip(L1,L2):
		outS += ('%-' + str(max_width) + 's' + spacing + l2 + '\n') % l1

	if auto_print: wuml.jupyter_print(outS, display_all_rows=True, display_all_columns=True)
	else: return outS


def block_matrix_concatenate(matrix_list, spacing='\t\t'):

	strList = []
	for m in matrix_list:
		#strList.append(pretty_np_array(m).strip())
		strList.append(pretty_np_array(m))

	return block_string_concatenate(strList, spacing=spacing)

def block_string_concatenate(strList, spacing='\t'):
	all_concat = ''
	for single_block in strList:

		if all_concat == '':
			all_concat = single_block
		else:
			all_concat = block_two_string_concatenate(all_concat, single_block, spacing=spacing)

	return all_concat

if __name__ == "__main__":
	str1 = 'Hello world\nA interesting string'
	str2 = 'Second Block\nto the right\nTooLong'
	outS = block_two_string_concatenate(str1, str2)
	print(outS)

	str1 = 'Hello world\nA interesting string'
	str2 = 'Second Block, to the right, TooLong'
	str3 = 'Third Block, \n make it shorter'
	all_3 = [str1, str2, str3]
	outS = block_string_concatenate(all_3)
	print(outS)


