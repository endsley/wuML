#!/usr/bin/env python

from wplotlib import histograms
import numpy as np
import wuml
import torch

def pretty_np_array(m, front_tab='', verticalize=False):
	if type(m) == type(torch.tensor([])):
		m = m.cpu().detach().numpy()

	if verticalize:
		if len(m.shape) == 1:
			m = np.atleast_2d(m).T

	out_str = front_tab + str(m).replace('\n ','\n' + front_tab).replace('[[','[').replace(']]',']') + '\n'
	#out_str = str(out_str).replace('. ',' ')
	out_str = str(out_str).replace('.]',']')
	return out_str

def block_two_string_concatenate(str1, str2, spacing='\t'):
	L1 = str1.split('\n')
	L2 = str2.strip().split('\n')

	if len(L1) > len(L2):
		Δ = len(L1) - len(L2)
		for ι in range(Δ):
			L2.append('\n')

	max_width = len(max(L1, key=len))
	outS = ''
	for l1, l2 in zip(L1,L2):
		outS += ('%-' + str(max_width) + 's' + spacing + l2 + '\n') % l1

	return outS


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


