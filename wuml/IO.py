#!/usr/bin/env python

import os 
import sys
import types
import torch
import pickle
import numpy as np
import wuml
import wplotlib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

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

def clear_current_line():
	sys.stdout.write("\r")
	sys.stdout.write("\033[K")
	sys.stdout.flush()

def write_to_current_line(txt):
	clear_current_line()
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

	output = wuml.pretty_np_array(np.hstack((col_1, col_2)))
	if labels is not None:
		output = wuml.pretty_np_array(labels) + output

	return output

def output_regression_result(y, ŷ, write_path=None):
	y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
	ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=2))

	if y.shape[0] == 1: y = y.T
	if ŷ.shape[0] == 1: ŷ = ŷ.T
	
	# Draw Histogram
	Δy = np.absolute(ŷ - y)
	avg_Δ = 'Avg error: %.4f\n\n'%(np.sum(Δy)/Δy.shape[0])

	H = wplotlib.histograms()
	H.histogram(Δy, num_bins=15, title='Histogram of Errors', 
				xlabel='Error Amount', ylabel='Error count',
				facecolor='blue', α=0.5, path=None)


	A = wuml.pretty_np_array(np.array([['y', 'ŷ']]))
	B = wuml.pretty_np_array(np.hstack((y, ŷ)))
	C = avg_Δ + A + B

	if write_path is not None: wuml.write_to(C, write_path)

	return C


class summarize_regression_result:
	def __init__(self, y, ŷ):
		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
		ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=2))
	
		if y.shape[0] == 1: y = y.T
		if ŷ.shape[0] == 1: ŷ = ŷ.T
		
		self.y = y
		self.ŷ = ŷ
		self.side_by_side_Y = np.hstack((self.y, self.ŷ))

		self.Δy = np.absolute(self.ŷ - self.y)

	def avg_error(self):
		Δy = self.Δy
		avg_Δ = np.sum(Δy)/Δy.shape[0]
		return avg_Δ

	def error_histogram(self):
		Δy = self.Δy
		avg_Δ = 'Avg error: %.4f\n\n'%(np.sum(Δy)/Δy.shape[0])
	
		H = wplotlib.histograms()
		H.histogram(Δy, num_bins=15, title='Histogram of Errors', 
					xlabel='Error Amount', ylabel='Error count',
					facecolor='blue', α=0.5, path=None)
		
	def true_vs_predict(self, write_path=None, sort_based_on_label=False, print_result=False):
		A = wuml.pretty_np_array(np.array([['y', 'ŷ']]))

		if sort_based_on_label:
			Yjoing = np.hstack((self.y, self.ŷ))
			sorted_df = wuml.sort_matrix_rows_by_a_column(Yjoing, 0)
			B = wuml.pretty_np_array(sorted_df.values)
		else: B = wuml.pretty_np_array(np.hstack((self.y, self.ŷ)))
		avg_Δ = 'Avg error: %.4f\n\n'%(self.avg_error())
		C = avg_Δ + A + B
	
		if print_result: print(C)
		if write_path is not None: wuml.write_to(C, write_path)
		return C
	

class summarize_classification_result:
	def __init__(self, y, ŷ):
		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
		ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=2))
	
		if y.shape[0] == 1: y = y.T
		if ŷ.shape[0] == 1: ŷ = ŷ.T
		
		self.y = y
		self.ŷ = ŷ
		self.side_by_side_Y = np.hstack((self.y, self.ŷ))

		self.Δy = np.absolute(self.ŷ - self.y)

	def avg_error(self):
		Acc= accuracy_score(self.y, self.ŷ)
		return Acc

	def error_histogram(self):
		pass

		#Δy = self.Δy
		#avg_Δ = 'Avg error: %.4f\n\n'%(np.sum(Δy)/Δy.shape[0])
	
		#H = wplotlib.histograms()
		#H.histogram(Δy, num_bins=15, title='Histogram of Errors', 
		#			xlabel='Error Amount', ylabel='Error count',
		#			facecolor='blue', α=0.5, path=None)
		
	def true_vs_predict(self, write_path=None, sort_based_on_label=False, print_result=False):
		A = wuml.pretty_np_array(np.array([['y', 'ŷ']]))

		if sort_based_on_label:
			Yjoing = np.hstack((self.y, self.ŷ))
			sorted_df = wuml.sort_matrix_rows_by_a_column(Yjoing, 0)
			B = wuml.pretty_np_array(sorted_df.values)
		else: B = wuml.pretty_np_array(np.hstack((self.y, self.ŷ)))
		avg_Δ = 'Avg error: %.4f\n\n'%(self.avg_error())
		C = avg_Δ + A + B
	
		if print_result: print(C)
		if write_path is not None: wuml.write_to(C, write_path)
		return C
	

