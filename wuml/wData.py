
import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


from sklearn import preprocessing
import wuml

from wuml.type_check import *
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class wData:
	def __init__(self, xpath=None, ypath=None, column_names=None, path_prefix='',
					label_column_name=None, label_column_id=None,
					dataFrame=None, X_npArray=None, Y_npArray=None, extra_data=None,
					first_row_is_label=False, 
					row_id_with_feature_names=None, first_column_as_sample_index=False, 
					label_type=None, label2_type=None, #it should be either 'continuous' or 'discrete'
					encode_discrete_label_to_one_hot=False,
					xtorchDataType=torch.FloatTensor, ytorchDataType=torch.FloatTensor, 
					batch_size=32, randomly_shuffle_batch=True, columns_to_ignore=None,
					replace_this_entry_with_nan=None, preprocess_data=None):
		'''
			first_row_is_label :  True of False
			row_id_with_feature_names: None, if the label is not the first row 0, set with this
			dataFrame: if dataFrame is set, it ignores the path and use the dataFrame directly as the data itself
			ypath: loads the data as the label
			label_column_name: if the label is loaded together with xpath, this separates label into Y
			preprocess_data: 'center and scale', 'linearly between 0 and 1', 'between 0 and 1 via cdf'
			first_column_as_sample_index: if the first column is used as sample ID
			extra_data: is a list of data that can be appended to the dataset, it has to be the same number of samples as X and y
			path_prefix: string or list, if list, automatically goes through potential folders where the data might be
		'''
		self.label_column_name = label_column_name
		self.label_column_id = label_column_id

		self.randomly_shuffle_batch = randomly_shuffle_batch

		if dataFrame is not None:
			self.df = dataFrame

			if first_row_is_label: 
				self.df = self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0])
	
			if first_column_as_sample_index: 
				self.df = self.df.set_index(list(self.df)[0])

		elif X_npArray is not None:
			if wtype(column_names) == 'str': column_names = [column_names]
			self.df = pd.DataFrame(X_npArray, columns=column_names)
			if first_row_is_label: 
				self.df = self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0])
	
			if first_column_as_sample_index: 
				self.df = self.df.set_index(list(self.df)[0])

		else:
			pth = wuml.append_prefix_to_path(path_prefix, xpath)
			self.df = pd.read_csv (pth, header=None)
			if first_column_as_sample_index: first_column_as_sample_index = 0
			if first_row_is_label: 
				self.df = pd.read_csv (pth, header=0, index_col=first_column_as_sample_index)
			else:
				self.df = pd.read_csv (pth, header=row_id_with_feature_names, index_col=first_column_as_sample_index)



		if replace_this_entry_with_nan is not None:
			self.df = self.df.replace(replace_this_entry_with_nan, np.nan)

		self.strip_white_space_from_column_names()
		self.Y = None
		self.extra_data = None		# if there exists a 2nd label

#		if Y_npArray is not None:
#			self.Y = Y_npArray
#			if encode_discrete_label_to_one_hot:
#				self.Y = wuml.one_hot_encoding(self.Y)
#
#		elif ypath is not None: 
#			ypth = wuml.append_prefix_to_path(path_prefix, ypath)
#
#			if label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')
#			self.Y = np.loadtxt(ypth, delimiter=',', dtype=np.float32)			
#			if label_type == 'discrete': 
#				self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0
#				if encode_discrete_label_to_one_hot:
#					self.Y = wuml.one_hot_encoding(self.Y)
#
#		elif label_column_name is not None:
#			if label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')
#			self.Y = self.df[label_column_name].values
#			if label_type == 'discrete': 
#				self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0
#				if encode_discrete_label_to_one_hot:
#					self.Y = wuml.one_hot_encoding(self.Y)
#
#			self.delete_column(label_column_name)
#		elif label_column_id is not None:
#			if label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')
#			self.Y = self.df[label_column_id].values
#			if label_type == 'discrete': 
#				self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0
#				if encode_discrete_label_to_one_hot:
#					self.Y = wuml.one_hot_encoding(self.Y)
#
#			self.delete_column(label_column_id)



		if columns_to_ignore is not None: self.delete_column(columns_to_ignore)

		self.X = self.df.values
		self.label_type = label_type
		self.shape = self.df.shape
		self.batch_size = batch_size
		self.torchloader = None
		self.columns = self.df.columns
		self.path_prefix = path_prefix


		self.format_label(Y_npArray=Y_npArray, ypath=ypath, label_column_name=label_column_name, label_column_id=label_column_id, encode_discrete_label_to_one_hot=encode_discrete_label_to_one_hot)


		self.Data_preprocess(preprocess_data)
		self.initialize_pytorch_settings(xtorchDataType, ytorchDataType)

		self.check_for_missingness()


	def format_label(self, Y_npArray=None, ypath=None, label_column_name=None, label_column_id=None, encode_discrete_label_to_one_hot=False):

		# as long as label is needed, we must designate continuous or discrete labels
		if not np.all(np.array([Y_npArray, ypath, label_column_name, label_column_id]) == None):
			if self.label_type is None: raise ValueError('If you are using labels, you must include the argument label_type= "continuout" or "discrete"')

		if Y_npArray is not None:
			self.Y = Y_npArray
		elif ypath is not None: 
			ypth = wuml.append_prefix_to_path(self.path_prefix, ypath)
			self.Y = np.loadtxt(ypth, delimiter=',', dtype=np.float32)			
		elif label_column_name is not None:
			self.Y = self.df[label_column_name].values
			self.delete_column(label_column_name)
		elif label_column_id is not None:
			self.Y = self.df[label_column_id].values
			self.delete_column(label_column_id)

		if self.label_type == 'discrete': 
			self.Y = LabelEncoder().fit_transform(self.Y)	#Make sure label start from 0
			if encode_discrete_label_to_one_hot:
				self.Y = wuml.one_hot_encoding(self.Y)

	def check_for_missingness(self):
		# raise a warning if there are missing entries within the data
		try:
			mL = np.sum(np.isnan(self.X))
			if mL > 0: wuml.jupyter_print('\nWarning: %d entries are missing.'%(mL))
		except: pass

		if self.Y is None: return
		missingLabels = wuml.identify_missing_labels(self.Y)
		if missingLabels is not None: 
			if missingLabels > 0: 
				mL = np.sum(np.isnan(self.Y))
				wuml.jupyter_print('Warning: %.5f percent or %d samples of the labels are missing:  \n'%(missingLabels*100, mL))
				removeSample = input("Would you like to remove the samples with missing labels [y]/n?")
				if removeSample == '' or removeSample == 'y' or removeSample == 'Y':
					wuml.remove_rows_with_missing_labels(self)


	def Data_preprocess(self, preprocess_data='center and scale'):
		#	Various ways to preprocess the data
		if preprocess_data == 'center and scale':
			self.X = preprocessing.scale(self.X)
			self.df = pd.DataFrame(data=self.X, columns=self.df.columns)
		elif preprocess_data == 'linearly between 0 and 1':
			self.X = wuml.wuml.map_data_between_0_and_1(self.X, output_type_name='ndarray', map_type='linear')
			self.df = pd.DataFrame(data=self.X, columns=self.df.columns)
		elif preprocess_data == 'between 0 and 1 via cdf':
			self.X = wuml.wuml.map_data_between_0_and_1(self.X, output_type_name='ndarray', map_type='cdf')
			self.df = pd.DataFrame(data=self.X, columns=self.df.columns)

	def initialize_pytorch_settings(self, xtorchDataType, ytorchDataType):
		#	Property set the data type in Pytorch
		self.xtorchDataType = xtorchDataType				
		if self.label_type == 'discrete': 
			self.ytorchDataType = torch.LongTensor
		else:
			self.ytorchDataType = ytorchDataType				

		if torch.cuda.is_available(): self.device = 'cuda'
		else: self.device = 'cpu'

	def swap_label(self, column_name_use_for_label):
		labelC = wuml.ensure_DataFrame(self.Y, columns=self.label_column_name)
		newLabel = self.pop_column(column_name_use_for_label)
		self.replace_label(newLabel, label_name=column_name_use_for_label)
		self.append_columns(labelC)

	def replace_label(self, newY, label_name=None):
		self.Y = ensure_numpy(newY)
		if label_name is not None: self.label_column_name = label_name

	def strip_white_space_from_column_names(self):
		# make sure to strip white space from column names
		update_col_names = []
		for i, col_name in enumerate(self.df.columns):
			if type(self.df.columns[i]).__name__ == 'str':
				update_col_names.append(col_name.strip())
			else:
				update_col_names.append(col_name)

		self.df.columns = update_col_names

	def sort_by(self, column_name, ascending=True):
		self.df = self.df.sort_values(column_name, ascending=ascending)
		self.update_DataFrame(self.df)

	def get_column_names_as_a_list(self):
		return ensure_wData(self.df.columns)

	def get_columns(self, columns):
		if type(columns).__name__ == 'int': 
			return ensure_wData(self.df.iloc[:,columns], column_names=[columns])
			
		columns = ensure_list(columns)
		subColumns = self.df[columns]
		return ensure_wData(subColumns)

	def update_data(self, data, columns=None):
		if wtype(data) == 'DataFrame':
			self.update_DataFrame(data)
		elif wtype(data) == 'ndarray':
			new_df = pd.DataFrame(data, columns=columns)
			self.update_DataFrame(new_df)
		else:
			raise ValueError('data_loading.update_data function does not recognizes input data type = %s', wtype(data))


	def update_DataFrame(self, df):
		self.df = df
		self.columns = self.df.columns
		self.X = self.df.values
		self.shape = self.df.shape
	
	def reset_index(self):
		self.df.reset_index(drop=True, inplace=True)	
		self.update_DataFrame(self.df)

	def append_rows(self, new_data, reset_index=True):
		df = ensure_DataFrame(new_data, columns=self.columns)
		self.update_DataFrame(pd.concat([self.df,df], axis=0))
		if reset_index: self.reset_index()


	def append_columns(self, new_data, column_names=None):
		df = ensure_DataFrame(new_data, columns=column_names)
		self.update_DataFrame(pd.concat([self.df,df], axis=1))

	def rename_columns(self, column_names):
		if type(column_names).__name__ == 'str':
			column_names = [column_names]

		column_names = np.squeeze(ensure_numpy(column_names)).tolist()
		self.df.columns = column_names
		self.columns = column_names

	def pop_column(self, column_name):
		C = self.get_columns(column_name)
		self.delete_column(column_name)
		return C

	def delete_column(self, column_name):
		if type(column_name) == type([]):
			for name in column_name:
				if name in self.df.columns:
					del self.df[name]

		elif type(column_name) == type(''):
			if column_name in self.df.columns:
				del self.df[column_name]

		elif type(column_name) == type(0):
			if column_name in self.df.columns:
				del self.df[column_name]

		self.columns = self.df.columns
		self.update_DataFrame(self.df)

	def info(self):
		wuml.jupyter_print(self.df.info())

	def get_data_as(self, data_type): #'DataFrame', 'read_csv', 'Tensor'
		if data_type == 'wData': return self
		if data_type == 'Tensor': 
			x = torch.from_numpy(self.df.values)
			x = Variable(x.type(self.xtorchDataType), requires_grad=False)
			X = x.to(self.device, non_blocking=True )
			return X
		if data_type == 'DataFrame': return self.df
		if data_type == 'ndarray': 
			#self.df.values[subset]
			return self.df.values
		if data_type == 'DataLoader':		# and self.torchloader is None 
			self.DM = wuml.DManager(self.df.values, self.Y, self.extra_data)
			self.torchloader = DataLoader(dataset=self.DM, batch_size=self.batch_size, shuffle=self.randomly_shuffle_batch, pin_memory=True, num_workers=1)
			return self.torchloader

	def get_all_samples_from_a_class(self, class_name_or_id):
		class_samples = np.empty((0,self.X.shape[1]))
		for i, j in enumerate(self.Y):
			if j == class_name_or_id:
				class_samples = np.vstack((class_samples, self.X[i]))

		return class_samples

	def retrieve_scalar_value(self):
		return self.df.to_numpy()[0,0]

	def to_csv(self, path, add_row_indices=False, include_column_names=True, float_format='%.4f'):
		LCn = self.label_column_name

		if LCn is not None:
			self.df[LCn] = self.Y
			self.df.to_csv(path, index=add_row_indices, header=include_column_names, float_format=float_format)
			self.delete_column(LCn)
		elif self.Y is not None:
			if self.label_column_name is not None:
				self.df[self.label_column_name] = self.Y
				self.df.to_csv(path, index=add_row_indices, header=include_column_names, float_format=float_format )
				self.delete_column(self.label_column_name)
			else:
				self.df['label'] = self.Y
				self.df.to_csv(path, index=add_row_indices, header=include_column_names, float_format=float_format )
				self.delete_column('label')
		else:
			self.df.to_csv(path, index=add_row_indices, header=include_column_names, float_format=float_format)

	def __getitem__(self, item):
		#	If item is string, it will return the column corresponding to the column name
		#	If item is int, it will return the row

		if type(item).__name__ == 'str': 
			return self.get_columns(item)
		elif type(item).__name__ == 'tuple': 
			return ensure_wData(self.df.iloc[item])
		else:
			raise ValueError('Error: recognized input, wData must be [0:3] or [0:2, 0:3] format. You can also get the column by string')

	def __str__(self): 
		return str(self.df)

	def __repr__(self): 
		return str(self.df)

	def __iter__(self): 
		self.itr_count = 0
		return self

	def __next__(self):
		''''Returns the next value from team object's lists '''
		try:
			nextItm = ensure_wData(self.df.iloc[self.itr_count].to_frame().transpose())
			#nextItm = self[self.itr_count]
			self.itr_count += 1
			return nextItm
		except:
			self.itr_count = 0
			raise StopIteration


	def plot_2_columns_as_scatter(self, column1, column2):
		if wuml.get_commandLine_input()[1] == 'disabled': return
		X = self.df[column1].to_numpy()
		Y = self.df[column2].to_numpy()
		
		#lp = wuml.scatter(figsize=(10,5))		# (width, height)
		#lp.plot_scatter(X, Y, column1 + ' vs ' + column2, column1, column2, ticker_fontsize=8 )

		lp = wuml.scatter(X, Y, title=str(column1) + ' vs ' + str(column2), 
				xlabel=str(column1), ylabel=str(column2), ticker_fontsize=8)
