
import pandas as pd
import numpy as np

class load_csv:
	def __init__(self, xpath=None, ypath=None, dataFrame=None, row_id_with_label=None , sample_id_included=False):
		'''
			row_id_with_label :  None = no labels, 0 = top row is the label
		'''
		if dataFrame is not None:
			self.df = dataFrame
		else:
			self.df = pd.read_csv (xpath, header=row_id_with_label)

		self.shape = self.df.shape

	def get_data_as(self, data_type): #'DataFrame', 'read_csv', 'Tensor'
		if data_type == 'read_csv': return self
		if data_type == 'DataFrame': return self.df



	def __getitem__(self, item):
		return self.df.values[item]

	def __str__(self): 
		print(self.df)
 
