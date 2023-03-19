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
import marshal, types
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from IPython.display import clear_output, HTML, Math
from wuml.type_check import *
from IPython.display import display_html
from itertools import chain,cycle
  


#	display 2 dataframes side by side
def DF_display_side_by_side(*args,titles=cycle([''])):
	#	Example usage
	#df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns=['A','B','C','D',])
	#df2 = pd.DataFrame(np.arange(16).reshape((4,4)),columns=['A','B','C','D',])
	#display_side_by_side(df1,df2,df1, titles=['Foo','Foo Bar']) 

	html_str=''
	for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
		html_str+='<th style="text-align:center"><td style="vertical-align:top">'
		html_str+=f'<h2 style="text-align: center;">{title}</h2>'
		html_str+=df.to_html().replace('table','table style="display:inline"')
		html_str+='</td></th>'
	display_html(html_str,raw=True)


def print_two_matrices_side_by_side(M1, M2, title1=None, title2=None, auto_print=True, rounding=3):

	if wuml.isnotebook():			# and wtype(M1) =='DataFrame' and wtype(M2) =='DataFrame':
		M1 = ensure_DataFrame(M1)
		M2 = ensure_DataFrame(M2)
		wuml.DF_display_side_by_side(M1, M2,titles=[title1, title2])
	else:
		eK = wuml.pretty_np_array(M1, front_tab='', title=title1, auto_print=False)
		eQ = wuml.pretty_np_array(M2, front_tab='\t', title=title2, auto_print=False)
		wuml.block_two_string_concatenate(eK, eQ, spacing='\t\t', add_titles=[], auto_print=auto_print)



def jupyter_print(value, display_all_rows=False, display_all_columns=False, font_size=1, latex=False, endString=None, testing_mode=''):
	cmds = wuml.get_commandLine_input()
	if testing_mode == 'disabled' or cmds[1] == 'disabled': return

	#font_size is from 1 to 6
	font_size = int(6 - font_size)
	if font_size > 6: font_size = 6
	if font_size < 0: font_size = 1

	if wtype(value) == 'result_table': 
		value = value.df

	if display_all_columns: pd.set_option('display.max_columns', None) 
	if display_all_rows: pd.set_option('display.max_rows', None)
	if wuml.isnotebook():
		if wtype(value) == 'DataFrame': 
			display(value)
		elif wtype(value) == 'ndarray': 
			display(ensure_DataFrame(value))
		elif wtype(value) == 'wData': 
			display(value.df)
		elif wtype(value) == 'Index': 
			value = str(value.tolist())
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wtype(value) == 'tuple': 
			value = str(value)
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wtype(value) == 'str': 
			#value = value.replace('\r','<br>')
			#value = value.replace('\n','<br>')
			#value = value.replace('\t','&nbsp &nbsp')
			if latex:
				display(Math(r'%s'%value))
			else:
				print(value)
				#wuml.write_to_current_line(value)
				#str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
				#display(HTML(data=str2html))
		else:
			if wtype(value) == 'Tensor': value = wuml.ensure_numpy(value)
			print(value)

		pd.set_option('display.max_rows', 10)
	else:
		if wtype(value) == 'Tensor': value = wuml.ensure_numpy(value)
		print(value)
	if endString is not None: print(endString)

def set_terminal_print_options(precision=3):
	np.set_printoptions(precision=precision)
	np.set_printoptions(linewidth=400)
	np.set_printoptions(suppress=True)

	torch.set_printoptions(edgeitems=3)
	torch.set_printoptions(sci_mode=False)
	torch.set_printoptions(precision=precision)
	torch.set_printoptions(linewidth=400)

	pd.set_option('display.width', 1000)
	pd.set_option('display.max_columns', 80)
	#pd.set_option("display.max_rows", 500)
	pd.set_option("display.precision", precision)

def csv_load(xpath, ypath=None, shuffle_samples=False):
	X = np.genfromtxt(	xpath, delimiter=',')
	Y = None
	if ypath is not None: 
		Y = np.genfromtxt(ypath, delimiter=',')

	if shuffle_samples:
		if Y is None:
			X = shuffle(X, random_state=0)
		else:
			X, Y = shuffle(X, Y, random_state=0)

	if Y is None: return X
	else: return [X,Y]


def csv_out(X, path, add_row_indices=False, include_column_names=False, float_format='%.4f'):
	X = wuml.ensure_wData(X)
	X.to_csv(path, add_row_indices=add_row_indices, include_column_names=include_column_names, float_format=float_format)


def label_csv_out(y, path, float_format='%.4f'):
	np.savetxt(path, y, delimiter=',', fmt=float_format) 


def read_txt_file(path):
	f = open(path, 'r')
	content = f.read()
	f.close()
	return content

def save_torch_network(network_obj, path):
	N = network_obj
	netData  = N.output_network_data_for_storage()
	pickle_dump(netData, path)

def load_torch_network(path, load_as_cpu_or_gpu=None): # load_as_cpu_or_gpu, if set to 'cpu' or 'cuda', it will try to force it 
	if load_as_cpu_or_gpu is not None:
		wuml.pytorch_device = load_as_cpu_or_gpu

	netData = pickle_load(path)
	
	if netData['name'] == 'basicNetwork':
		net = wuml.basicNetwork(None, None, pickled_network_info=netData)
	elif netData['name'] == 'combinedNetwork':
		code = marshal.loads(netData['network_behavior_on_call'])
		network_behavior_on_call = types.FunctionType(code, globals(), "network_behavior_on_call")

		code = marshal.loads(netData['costFunction'])
		costFunction = types.FunctionType(code, globals(), "costFunction")
		net = wuml.combinedNetwork(None, None, None, costFunction, network_behavior_on_call, pickled_network_info=netData, data_mean=netData['μ'], data_std=netData['σ'])

		if 'explainer' in netData.keys():
			stored_reference_data = wuml.ensure_wData(netData['reference_data'] , column_names=netData['column_names'])
			net.explainer = wuml.explainer(stored_reference_data, net, explainer_algorithm=netData['explainer_algorithm'], which_model_output_to_use=netData['which_model_output_to_use'])


	elif netData['name'] == 'autoencoder':
		code = marshal.loads(netData['costFunction'])
		costFunction = types.FunctionType(code, globals(), "costFunction")
		net = wuml.autoencoder(None, None, costFunction=costFunction, pickled_network_info=netData)
	else:
		raise ValueError('\n\tError loading torch network: %s is unrecognized network class'%netData['name'])

	return net

#
#func(10)  # gives 100



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

def list_all_files_in_directory(path, with_extension=None):
	all_files = os.listdir(path)

	if with_extension is not None:
		new_all_files = []
		for f in all_files:
			file_extension = Path(f).suffix
			if with_extension == file_extension:
				new_all_files.append(f)

		all_files = new_all_files

	return all_files

def ensure_path_exists(path):
	if os.path.exists(path): return True
	os.mkdir(path)
	return False

def path_list_exists(path_list):
	for i in path_list:
		if os.path.exists(i) == False: 
			return False

	return True

# path_prefix can be a string or a list of string
# if list, then it will check the existence given the prefix
def append_prefix_to_path(path_prefix, fname):
	if wtype(path_prefix) == 'str':
		ypth = path_prefix + fname
	elif wtype(path_prefix) == 'list':
		for pre in path_prefix:
			ypth = pre + fname
			if wuml.path_exists(ypth):
				break
	else:
		raise ValueError('path_prefix must be a string or a list of strings')

	return ypth


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
	if get_commandLine_input()[1] == 'disabled': return
	clear_current_line()
	write_to_current_line('Completion Status: %.3f'%percent)
	sys.stdout.flush()

def clear_current_line():
	if get_commandLine_input()[1] == 'disabled': return
	#print(..., end="")
	#sys.stdout.write('\033[2K\033[1G')
	sys.stdout.write("\r")		#jump to the beginning of line
	sys.stdout.write("\033[K")	#erase to the end of line
	#sys.stdout.write("\033[F")		move cursor to beginning of previous line
	#“\033[A” – move cursor up one line.
	
	sys.stdout.flush()

def write_to_current_line(txt):
	if get_commandLine_input()[1] == 'disabled': return

	clear_current_line()
	clear_output(wait = True)
	sys.stdout.write(txt)
	sys.stdout.flush()

def clear_previous_line(num_of_lines=None):
	if get_commandLine_input()[1] == 'disabled': return
	if num_of_lines is None: num_of_lines = 1
	for i in range(num_of_lines):
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
		wuml.jupyter_print(output)

		return output


def get_commandLine_input(num_of_args=10):
	command_list = []	
	for i in range(num_of_args):
		try:
			command_list.append(sys.argv[i])
		except:
			command_list.append('')

	return command_list

def output_regression_result(y, ŷ, write_path=None, sort_by='none', ascending=False, print_out=['histograms', 'mean absolute error', 'true label vs prediction table']):
	y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
	ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=2))

	if y.shape[0] == 1: y = y.T
	if ŷ.shape[0] == 1: ŷ = ŷ.T
	
	# Draw Histogram
	Δy = np.absolute(ŷ - y)

	if 'mean absolute error' in print_out:
		avg_Δ = 'Avg error: %.4f'%(np.sum(Δy)/Δy.shape[0])
		jupyter_print(avg_Δ)

	if 'histograms' in print_out:
		wplotlib.histograms(Δy, num_bins=15, title='Histogram of Errors', xlabel='Error Amount', ylabel='Error count',
					facecolor='blue', α=0.5, path=None)

	df = pd.DataFrame(np.hstack((y, ŷ, Δy)), columns=['y', 'ŷ', 'Δy'])
	if sort_by == 'error':
		df = df.sort_values('Δy', ascending=ascending)
	elif sort_by == 'label':
		df = df.sort_values('y', ascending=ascending)
	elif sort_by == 'output':
		df = df.sort_values('ŷ', ascending=ascending)

	if not wuml.isnotebook():
		if 'true label vs prediction table' in print_out:
			jupyter_print(df, display_all_rows=True, display_all_columns=True)
	
	if write_path is not None: wuml.write_to(str(df), write_path)
	return wuml.ensure_wData(df)


class summarize_regression_result:
	def __init__(self, y, ŷ, print_out=['avg absolute error', 'true v predict'], rounding=3):
		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=rounding))
		ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=rounding))
	
		if y.shape[0] == 1: y = y.T
		if ŷ.shape[0] == 1: ŷ = ŷ.T
		
		self.y = y
		self.ŷ = ŷ
		self.Δy = np.round(np.absolute(self.ŷ - self.y), 3)
		self.mean_absolute_error = self.avg_error()

		if print_out is not None:
			if 'avg absolute error' in print_out:
				jupyter_print('Mean Absolute Error: %.4f'%self.mean_absolute_error)

			if 'true v predict' in print_out:
				self.true_vs_predict(print_out=True)

		self.check_for_errors(print_out)

	def check_for_errors(self, print_out):
		if print_out is None: return
		for i in print_out:
			if i not in ['avg absolute error', 'true v predict']:
				raise ValueError('\n\tError in summarize_regression_result: %s is not recognized print_out option'%i)


	def avg_error(self):
		Δy = self.Δy
		avg_Δ = np.sum(Δy)/Δy.shape[0]
		return avg_Δ

	def error_histogram(self):
		Δy = self.Δy
		avg_Δ = 'Avg error: %.4f\n\n'%(np.sum(Δy)/Δy.shape[0])
	
		H = wplotlib.histograms(Δy, num_bins=15, title='Histogram of Errors', 
					xlabel='Error Amount', ylabel='Error count',
					facecolor='blue', α=0.5, path=None)
		
	def true_vs_predict(self, write_path=None, sort_by='none', ascending=False, print_out=False):
		cnames = np.array(['y', 'ŷ', 'Δy'])
		Yjoin = np.hstack((self.y, self.ŷ, self.Δy))
		df = pd.DataFrame(Yjoin, columns=cnames)

		# round the data frame decimals correctly
		y_ndeci = len(str(np.squeeze(self.y)[0]).split('.')[1])
		ŷ_ndeci = len(str(np.squeeze(self.ŷ)[0]).split('.')[1])
		Δy_ndeci = len(str(np.squeeze(self.Δy)[0]).split('.')[1])
		df = df.round({'y':y_ndeci, 'ŷ':ŷ_ndeci, 'Δy':Δy_ndeci})

		if sort_by == 'error':
			df = df.sort_values('Δy', ascending=ascending)
		elif sort_by == 'label':
			df = df.sort_values('y', ascending=ascending)
		elif sort_by == 'output':
			df = df.sort_values('ŷ', ascending=ascending)

		if print_out:
			jupyter_print(df, display_all_rows=True, display_all_columns=True)

		return wuml.ensure_wData(df)


class summarize_classification_result:
	def __init__(self, y, ŷ, print_out=['accuracy', 'true v predict labels']):

		y = np.atleast_2d(wuml.ensure_numpy(y, rounding=2))
		ŷ = np.atleast_2d(wuml.ensure_numpy(ŷ, rounding=2))
	
		if y.shape[0] == 1: y = y.T
		if ŷ.shape[0] == 1: ŷ = ŷ.T

		self.y = y
		self.ŷ = ŷ
		self.side_by_side_Y = np.hstack((self.y, self.ŷ))
		self.Δy = np.absolute(self.ŷ - self.y)

		self.accuracy = self.get_accuracy()
		if len(np.unique(self.y)) == 2: 
			self.Precision = wuml.precision(self.y, self.ŷ)
			self.Recall = wuml.recall(self.y, self.ŷ)

		if wuml.get_commandLine_input()[1] == 'disabled': return
		# Printing out the result
		if print_out is not None:
			if 'accuracy' in print_out:
				jupyter_print('Classification Accuracy: %.4f'%self.accuracy)
	
			if is_binary_label(y):
				jupyter_print('Precision: %.4f (probability that a positive prediction is correct)'%self.Precision)
				jupyter_print('Recall: %.4f (probability that we catch a positive event)'%self.Recall)

			if 'true v predict labels' in print_out:
				self.true_vs_predict(print_out=True)

		self.check_for_errors(print_out)

	def check_for_errors(self, print_out):
		if print_out is None: return
		for i in print_out:
			if i not in ['accuracy', 'true v predict labels']:
				raise ValueError('Error in summarize_classification_result: %s is not recognized print_out option'%i)

	def get_accuracy(self):
		Acc= accuracy_score(self.y, self.ŷ)
		return Acc
		
	def true_vs_predict(self, add_to_label='', write_path=None, sort_by='none', ascending=False, print_out=False):
		cnames = np.array(['y'+add_to_label, 'ŷ'+ add_to_label, 'Δy'+add_to_label])
		Yjoin = np.hstack((self.y, self.ŷ, self.Δy))
		df = pd.DataFrame(Yjoin, columns=cnames)

		if sort_by == 'error':
			df = df.sort_values('Δy', ascending=ascending)
		elif sort_by == 'label':
			df = df.sort_values('y', ascending=ascending)
		elif sort_by == 'output':
			df = df.sort_values('ŷ', ascending=ascending)

		if print_out:
			jupyter_print(df, display_all_rows=True, display_all_columns=True)

		return wuml.ensure_wData(df)


	

class result_table:
	def __init__(self, column_names, data=None):
		self.column_names = column_names
		if data is None:
			self.df = pd.DataFrame(columns=column_names)
		else:
			self.df = pd.DataFrame(data, columns=column_names)


	def add_row(self, row_data):
		zP = zip(self.column_names, row_data)
		D = {}
		for i,j in zP:
			D[i] = j

		self.df = self.df.append(D, ignore_index = True)

	def __str__(self): 
		return str(self.df)


	def get_column(self, column):
		if type(column).__name__ == 'int': 
			return wuml.ensure_wData(self.df.iloc[:,column])
		
		return wuml.ensure_wData(self.df[column])

