#!/usr/bin/env python

from wplotlib import histograms
import numpy as np
import wuml
import torch


#str.ljust(s, width[, fillchar])
#str.rjust(s, width[, fillchar])
#str.center(s, width[, fillchar])

def pretty_np_array(m, front_tab='', verticalize=False, title=None, auto_print=False):
	m = str(m)

	if type(m) == type(torch.tensor([])):
		m = m.cpu().detach().numpy()

	if verticalize:
		if len(m.shape) == 1:
			m = np.atleast_2d(m).T

	out_str = front_tab + str(m).replace('\n ','\n' + front_tab).replace('[[','[').replace(']]',']') + '\n'
	out_str = str(out_str).replace('.]',']')

	if type(title).__name__ == 'str':
		L1 = out_str.split('\n')
		L1_max_width = len(max(L1, key=len))
		t1 = str.center(title, L1_max_width)
		out_str = t1 + '\n' + out_str

	if auto_print: print(out_str)
	else: return out_str

def block_two_string_concatenate(str1, str2, spacing='\t', add_titles=[], auto_print=False):
	str1 = str(str1)
	str2 = str(str2)

	L1 = str1.split('\n')
	L2 = str2.strip().split('\n')

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

	if auto_print: print(outS)
	else: return outS


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


