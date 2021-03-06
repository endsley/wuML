#!/usr/bin/env python

import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import types
import torch
import pickle
import numpy as np
import wuml

import wplotlib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from IPython.display import clear_output, HTML, Math

def jupyter_print(value, display_all_rows=False, display_all_columns=False, font_size=3, latex=False):
	#font_size is from 1 to 6
	font_size = int(6 - font_size)
	if font_size > 6: font_size = 6
	if font_size < 0: font_size = 1

	if wuml.isnotebook():
		if wuml.wtype(value) == 'DataFrame': 
			if display_all_rows: pd.set_option('display.max_rows', None)
			if display_all_columns: pd.set_option('display.max_columns', None) 
			display(value)
		elif wuml.wtype(value) == 'wData': 
			if display_all_rows: pd.set_option('display.max_rows', None)
			if display_all_columns: pd.set_option('display.max_columns', None)
			display(value.df)
		elif wuml.wtype(value) == 'Index': 
			value = str(value.tolist())
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wuml.wtype(value) == 'tuple': 
			value = str(value)
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wuml.wtype(value) == 'str': 
			value = value.replace('\r','<br>')
			value = value.replace('\n','<br>')
			if latex:
				display(Math(r'%s'%value))
			else:
				str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
				display(HTML(data=str2html))
		else:
			print(value)

		pd.set_option('display.max_rows', 10)
	else:
		print(value)



def set_terminal_print_options(precision=3):
	np.set_printoptions(precision=precision)
	np.set_printoptions(linewidth=400)
	np.set_printoptions(suppress=True)

	torch.set_printoptions(edgeitems=3)
	torch.set_printoptions(sci_mode=False)
	torch.set_printoptions(precision=precision)
	torch.set_printoptions(linewidth=400)

	pd.set_option("display.max_rows", 500)
	pd.set_option("precision", precision)

def csv_out(X, path, add_row_indices=False, include_column_names=False, float_format='%.4f'):
	X = wuml.ensure_wData(X)
	X.to_csv(path, add_row_indices=add_row_indices, include_column_names=include_column_names, float_format=float_format)

def read_txt_file(path):
	f = open(path, 'r')
	content = f.read()
	f.close()
	return content

def pickle_dump(obj, path):
	pickle.dump( obj, open( path, "wb" ) )
   
def pickle_load(pth):
	return pickle.load( open( pth, "rb" ) )

def copy_file(src, destination):
	copyfile(src, destination)

def write_to(txt, path):
	f = open(path, 'w')
	f.write(txt)
	f.close()

def go_up_1_directory(full_path):	# path should be a string
	return str(Path(full_path).parents[0])

def initialize_empty_folder(path):
	ensure_path_exists(path)
	remove_files(path)

def path_exists(path):
	if os.path.exists(path): return True
	return False

def ensure_path_exists(path):
	if os.path.exists(path): return True
	os.mkdir(path)
	return False

def path_list_exists(path_list):
	for i in path_list:
		if os.path.exists(i) == False: 
			return False

	return True

def create_file(path):
	fin = open(path,'w')
	fin.close()

def delete_file(path):
	if os.path.exists(path):
		os.remove(path)

def remove_files(folder_path):	# make sure to end the path with /
	if not os.path.exists(folder_path): return 

	file_in_tmp = os.listdir(folder_path)
	for i in file_in_tmp:
		if os.path.isfile(folder_path + i):
			os.remove(folder_path + i)





#	Terminal Printing

def print_status(percent):
	clear_current_line()
	write_to_current_line('Completion Status: %.3f'%percent)
	sys.stdout.flush()

def clear_current_line():
	sys.stdout.write("\r")
	sys.stdout.write("\033[K")
	sys.stdout.flush()

def write_to_current_line(txt):
	clear_current_line()
	clear_output(wait = True)
	sys.stdout.write(txt)

def clear_previous_line():
	clear_current_line()
	sys.stdout.write("\r")
	sys.stdout.write("\033[F")
	sys.stdout.flush()
	clear_current_line()

def loss_optimization_printout(db, epoch, avgLoss, avgGrad, epoc_loop, slope):
	sys.stdout.write("\r\t\t%d/%d, MaxLoss : %f, AvgGra : %f, progress slope : %f" % (epoch, epoc_loop, avgLoss, avgGrad, slope))
	sys.stdout.flush()

def dictionary_to_str(dic):
	outstr = ''
	for i,j in dic.items():
		if type(j) == str: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == np.float64: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == bool: outstr += ('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == type: outstr += ('\t\t' + i + ' : ' + j.__name__ + '\n')
		elif type(j) == types.FunctionType: outstr += ('\t\t' + i + ' : ' + j.__name__ + '\n')
		elif type(j) == float: outstr += ('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == int: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		else:
			print('%s , %s is not recognized'%(i, str(type(j))))
			import pdb; pdb.set_trace()	

	return outstr

def output_two_columns_side_by_side(col_1, col_2, labels=None, rounding=3):
	col_1 = np.atleast_2d(wuml.ensure_numpy(col_1, rounding=rounding))
	col_2 = np.atleast_2d(wuml.ensure_numpy(col_2, rounding=rounding))

	if col_1.shape[0] == 1: col_1 = col_1.T
	if col_2.shape[0] == 1: col_2 = col_2.T

	if wuml.isnotebook():
		output = wuml.ensure_wData(np.hstack((col_1, col_2)), column_names=labels)
		jupyter_print(output, display_all_rows=True)
	else:
		output = wuml.pretty_np_array(np.hstack((col_1, col_2)))
		if labels is not None:
			output = wuml.pretty_np_array(labels) + output
		print(output)

		return output

def output_regression_result(y, ??, write_path=None):
	y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
	?? = np.atleast_2d(wuml.ensure_numpy(??, rounding=2))

	if y.shape[0] == 1: y = y.T
	if ??.shape[0] == 1: ?? = ??.T
	
	# Draw Histogram
	??y = np.absolute(?? - y)
	avg_?? = 'Avg error: %.4f\n\n'%(np.sum(??y)/??y.shape[0])

	H = wplotlib.histograms()
	H.histogram(??y, num_bins=15, title='Histogram of Errors', 
				xlabel='Error Amount', ylabel='Error count',
				facecolor='blue', ??=0.5, path=None)


	A = wuml.pretty_np_array(np.array([['y', '??']]))
	B = wuml.pretty_np_array(np.hstack((y, ??)))
	C = avg_?? + A + B

	if write_path is not None: wuml.write_to(C, write_path)

	return C


class summarize_regression_result:
	def __init__(self, y, ??):
		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
		?? = np.atleast_2d(wuml.ensure_numpy(??, rounding=2))
	
		if y.shape[0] == 1: y = y.T
		if ??.shape[0] == 1: ?? = ??.T
		
		self.y = y
		self.?? = ??
		self.side_by_side_Y = np.hstack((self.y, self.??))

		self.??y = np.absolute(self.?? - self.y)

	def avg_error(self):
		??y = self.??y
		avg_?? = np.sum(??y)/??y.shape[0]
		return avg_??

	def error_histogram(self):
		??y = self.??y
		avg_?? = 'Avg error: %.4f\n\n'%(np.sum(??y)/??y.shape[0])
	
		H = wplotlib.histograms()
		H.histogram(??y, num_bins=15, title='Histogram of Errors', 
					xlabel='Error Amount', ylabel='Error count',
					facecolor='blue', ??=0.5, path=None)
		
	def true_vs_predict(self, write_path=None, sort_based_on_label=False, print_result=False):
		#A = wuml.pretty_np_array(np.array([['y', '??']]))
		A = np.array([['y', '??']])
		if sort_based_on_label:
			Yjoing = np.hstack((self.y, np.round(self.??,3)))
			sorted_df = wuml.sort_matrix_rows_by_a_column(Yjoing, 0)
			#B = wuml.pretty_np_array(sorted_df.values)
			B = sorted_df.values
		else: 
			B = np.hstack((self.y, self.??))

		return wuml.ensure_wData(np.vstack((A,B)))


		#import pdb; pdb.set_trace()

		#avg_?? = 'Avg error: %.4f\n\n'%(self.avg_error())
		#C = avg_?? + A + B
	
		#if print_result: print(C)
		#if write_path is not None: wuml.write_to(C, write_path)

		#return ensure_wData(C, column_names=None)
		#return C
	

class summarize_classification_result:
	def __init__(self, y, ??):
		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
		?? = np.atleast_2d(wuml.ensure_numpy(??, rounding=2))
	
		if y.shape[0] == 1: y = y.T
		if ??.shape[0] == 1: ?? = ??.T
		
		self.y = y
		self.?? = ??
		self.side_by_side_Y = np.hstack((self.y, self.??))

		self.??y = np.absolute(self.?? - self.y)

	def avg_error(self):
		Acc= accuracy_score(self.y, self.??)
		return Acc

	def error_histogram(self):
		pass

		#??y = self.??y
		#avg_?? = 'Avg error: %.4f\n\n'%(np.sum(??y)/??y.shape[0])
	
		#H = wplotlib.histograms()
		#H.histogram(??y, num_bins=15, title='Histogram of Errors', 
		#			xlabel='Error Amount', ylabel='Error count',
		#			facecolor='blue', ??=0.5, path=None)
		
	def true_vs_predict(self, write_path=None, sort_based_on_label=False, print_result=False):
		A = wuml.pretty_np_array(np.array([['y', '??']]))

		if sort_based_on_label:
			Yjoing = np.hstack((self.y, self.??))
			sorted_df = wuml.sort_matrix_rows_by_a_column(Yjoing, 0)
			B = wuml.pretty_np_array(sorted_df.values)
		else: B = wuml.pretty_np_array(np.hstack((self.y, self.??)))
		avg_?? = 'Avg error: %.4f\n\n'%(self.avg_error())
		C = avg_?? + A + B
	
		if print_result: print(C)
		if write_path is not None: wuml.write_to(C, write_path)
		return C
	

