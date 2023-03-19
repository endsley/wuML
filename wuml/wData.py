
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
					dataFrame=None, X_npArray=None, Y_npArray=None, 
					extra_data=None, extra_data_preprocessing=None,
					first_row_is_label=False, 
					row_id_with_feature_names=None, first_column_as_sample_index=False, 
					label_type=None, label2_type=None, #it should be either 'continuous' or 'discrete'
					encode_discrete_label_to_one_hot=False,
					xtorchDataType=torch.FloatTensor, ytorchDataType=torch.FloatTensor, 
					columns_to_ignore=None, mv_columns_to_extra_data=None, only_keep_these_columns=None,
					batch_size=32, randomly_shuffle_batch=True, 
					replace_this_entry_with_nan=None, preprocess_data=None):
		'''
			first_row_is_label :  True of False
			row_id_with_feature_names: None, if the label is not the first row 0, set with this
			dataFrame: if dataFrame is set, it ignores the path and use the dataFrame directly as the data itself
			ypath: loads the data as the label
			label_column_name: if the label is loaded together with xpath, this separates label into Y
			preprocess_data: 'center and scale', 'linearly between 0 and 1', 'between 0 and 1 via cdf'
			first_column_as_sample_index: if the first column is used as sample ID
			extra_data: is a list of data that can be appended to the dataset, it has to be the same number of samples as X and y, if not a list, then it will be casted into a list
			extra_data_preprocessing is a list of lists including commands to preprocess, e.g., given 2 extra data [['center and scale'],['ensure_ proper discrete labels']]
			path_prefix: string or list, if list, automatically goes through potential folders where the data might be
			mv_columns_to_extra_data: a list of column names 
		'''
		self.path_prefix = path_prefix
		self.label_column_name = label_column_name
		
		self.label_column_id = label_column_id
		self.randomly_shuffle_batch = randomly_shuffle_batch
		self.row_id_with_feature_names = row_id_with_feature_names
		self.first_row_is_label = first_row_is_label
		self.replace_this_entry_with_nan = replace_this_entry_with_nan
		self.first_column_as_sample_index = first_column_as_sample_index
		self.xpath = xpath
		self.column_names = column_names
		self.only_keep_these_columns = only_keep_these_columns
		self.Y = None
		self.label_type = label_type
		self.batch_size = batch_size
		self.torchloader = None
		self.μ = None
		self.σ = None



		self.mv_columns_to_extra_data = mv_columns_to_extra_data
		self.extra_data_preprocessing = extra_data_preprocessing
		self.extra_data_dictionary = {}
		self.extra_data_dictionary['extra_data'] = extra_data
		self.extra_data_dictionary['numpy'] = self.xDat = []
		self.extra_data_dictionary['df'] = []


		# format the data
		self.format_data_based_on_input_type(X_npArray, dataFrame, columns_to_ignore)
		self.format_extra_data_based_on_input_type()
		self.format_label(Y_npArray=Y_npArray, ypath=ypath, label_column_name=label_column_name, label_column_id=label_column_id, encode_discrete_label_to_one_hot=encode_discrete_label_to_one_hot)
		if self.only_keep_these_columns is not None:
			retained_columns = self.get_columns(self.only_keep_these_columns)
			self.update_data(retained_columns)



		self.columns = self.column_names = self.df.columns		# you can use columns or column_names, they are exactly the same, columns_names is more descriptive, columns is compatible with pandas dataframe
		self.shape = self.df.shape

		self.Data_preprocess(preprocess_data)
		self.initialize_pytorch_settings(xtorchDataType, ytorchDataType)

		self.check_for_missingness()

	def format_extra_data_based_on_input_type(self):
		extra_data = self.extra_data_dictionary['extra_data']
		if extra_data is None and self.mv_columns_to_extra_data is None: return 
		if extra_data is None and self.mv_columns_to_extra_data is not None: extra_data = []								# make sure extra_data is a list 
		if wtype(self.mv_columns_to_extra_data) == 'str': self.mv_columns_to_extra_data = [self.mv_columns_to_extra_data]	# make sure it is in list format


		if wtype(extra_data) == 'str': 		#if string, it is the load path
			pth = wuml.append_prefix_to_path(self.path_prefix, extra_data)
			extra_data = pd.read_csv (pth, header=None)

		if wtype(extra_data) != 'list': 
			extra_data = [extra_data]

		if self.mv_columns_to_extra_data is not None:
			for col in self.mv_columns_to_extra_data:
				extra_data.append(self.pop_column(col))


		if wtype(self.extra_data_preprocessing) == 'str':
			self.extra_data_preprocessing = [[self.extra_data_preprocessing]]


		self.extra_data_dictionary['extra_data'] = extra_data
		for i, data in enumerate(extra_data):
			Dat_np = ensure_numpy(data)
			if self.extra_data_preprocessing is not None:
				for j in self.extra_data_preprocessing[i]:
					if j == 'center and scale':
						Dat_np = preprocessing.scale(Dat_np)
					elif j == 'ensure_ proper discrete labels':
						Dat_np = LabelEncoder().fit_transform(Dat_np)	#Make sure label start from 0

			self.extra_data_dictionary['numpy'].append(ensure_numpy(Dat_np))
			#self.extra_data_dictionary['df'].append(ensure_DataFrame(Dat_np))

	def round(self, rounding=3):
		self.df = self.df.round(decimals=rounding)
		self.update_data(self.df)

	def format_data_based_on_input_type(self, X_npArray, dataFrame, columns_to_ignore):
		first_row_is_label = self.first_row_is_label

		if dataFrame is not None:
			self.df = dataFrame

			if first_row_is_label: 
				self.df = self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0])
	
			if self.first_column_as_sample_index: 
				self.df = self.df.set_index(list(self.df)[0])

		elif X_npArray is not None:
			if wtype(self.column_names) == 'str': self.column_names = [self.column_names]
			self.df = pd.DataFrame(X_npArray, columns=self.column_names)
			if first_row_is_label: 
				self.df = self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0])
	
			if self.first_column_as_sample_index: 
				self.df = self.df.set_index(list(self.df)[0])

		else:
			pth = wuml.append_prefix_to_path(self.path_prefix, self.xpath)
			self.df = pd.read_csv (pth, header=None)
			if self.first_column_as_sample_index: self.first_column_as_sample_index = 0
			if first_row_is_label: 
				self.df = pd.read_csv (pth, header=0, index_col=self.first_column_as_sample_index)
			else:
				self.df = pd.read_csv (pth, header=self.row_id_with_feature_names, index_col=self.first_column_as_sample_index)

		if self.replace_this_entry_with_nan is not None:
			self.df = self.df.replace(self.replace_this_entry_with_nan, np.nan)

		self.strip_white_space_from_column_names()
		if columns_to_ignore is not None: self.delete_column(columns_to_ignore)

		self.X = self.df.values


	def format_label(self, Y_npArray=None, ypath=None, label_column_name=None, label_column_id=None, encode_discrete_label_to_one_hot=False):

		# as long as label is needed, we must designate continuous or discrete labels
		if not np.all(np.array([Y_npArray, ypath, label_column_name, label_column_id]) == None):
			if self.label_type is None: raise ValueError('\n\tError : If you are using labels, you must include the argument label_type= "continuout" or "discrete"')

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


	def Data_preprocess(self, preprocess_data='center and scale', mean=None, std=None):
		#	Various ways to preprocess the data
		if preprocess_data == 'center and scale':
			[self.X, self.μ, self.σ] = wuml.center_and_scale(self.X, return_type='ndarray', also_return_mean_and_std=True, mean=mean, std=std)
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
		self.device = wuml.get_current_device()

	def swap_label(self, column_name_use_for_label):
		labelC = wuml.ensure_DataFrame(self.Y, columns=self.label_column_name)
		newLabel = self.pop_column(column_name_use_for_label)
		self.replace_label(newLabel, label_name=column_name_use_for_label)
		self.append_columns(labelC)

	def replace_label(self, newY, label_name=None):
		self.Y = ensure_numpy(newY)
		self.Y = np.squeeze(self.Y)

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

	def sort_by(self, column_name, ascending=False):
		self.df = self.df.sort_values(column_name, ascending=ascending)
		self.update_DataFrame(self.df)

	def get_column_names_as_a_list(self):
		return ensure_wData(self.df.columns)

	def get_columns(self, columns):
		if type(columns).__name__ == 'int': 
			remaining_columns = ensure_wData(self.df.iloc[:,columns], column_names=[columns])
		else:	
			columns = ensure_list(columns)
			subColumns = self.df[columns]
			remaining_columns = ensure_wData(subColumns)

		remaining_columns.Y = self.Y
		remaining_columns.extra_data_dictionary = self.extra_data_dictionary
		return remaining_columns

	def update_data(self, data, columns=None):
		if wtype(data) == 'DataFrame':
			self.update_DataFrame(data)
		elif wtype(data) == 'wData':
			self.update_DataFrame(data.df)
		elif wtype(data) == 'ndarray':
			new_df = pd.DataFrame(data, columns=columns)
			self.update_DataFrame(new_df)
		else:
			raise ValueError('wData.update_data function does not recognizes input data type = %s'%wtype(data))


	def update_DataFrame(self, df):
		self.df = df
		self.columns = self.column_names = self.df.columns
		self.X = self.df.values
		self.shape = self.df.shape
	
	def update_column_type(self, update_dictionary):	# the dictionary looks like {'score': float, 'Accᶜ': float } with column name and type
		new_df = self.df.astype(update_dictionary)
		self.update_DataFrame(new_df)


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
		self.columns = self.column_names = column_names

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

		self.columns = self.column_names = self.df.columns
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
			self.DM = wuml.DManager(self.df.values, self.Y, self.extra_data_dictionary)
			self.torchloader = DataLoader(dataset=self.DM, batch_size=self.batch_size, shuffle=self.randomly_shuffle_batch, pin_memory=True, num_workers=1)
			return self.torchloader

	def get_all_samples_from_a_class(self, class_name_or_id):
		class_samples = np.empty((0,self.X.shape[1]))
		for i, j in enumerate(self.Y):
			if j == class_name_or_id:
				class_samples = np.vstack((class_samples, self.X[i]))

		class_samples = ensure_wData(class_samples, self.columns)
		N = class_samples.shape[0]
		class_samples.Y = np.full((N), class_name_or_id)
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
		elif type(item).__name__ == 'tuple' or type(item).__name__ == 'int' or type(item).__name__ == 'slice': 
			return ensure_wData(self.df.iloc[item])
		else:
			raise ValueError('Error: unrecognized input %s, wData must be [0:3] or [0:2, 0:3] format. You can also get the column by string'%type(item))

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
