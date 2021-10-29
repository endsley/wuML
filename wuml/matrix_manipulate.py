
import numpy as np
from wuml.type_check import *

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def double_center(Ψ):
	Ψ = ensure_numpy(Ψ)
	HΨ = Ψ - np.mean(Ψ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
	HΨH = (HΨ.T - np.mean(HΨ.T, axis=0)).T
	return HΨH

def get_np_column(data, column_id):
	X = ensure_numpy(data)
	return np.atleast_2d(X[:,column_id]).T

def sort_matrix_rows_by_a_column(X, column_names):
	df = ensure_DataFrame(X)
	sorted_df = df.sort_values(by=column_names)

	return sorted_df

def one_hot_encoding(Y, output_data_type='same'): 
	'''
		output_data_type : 'same' implies to use the same data type as input
							else use the data type you want, i.e., 'ndarray' , 'Tensor', 'DataFrame'
	'''
	Y1 = wuml.ensure_numpy(Y)

	Y1 = np.reshape(Y1,(len(Y1),1))
	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(Y1)

	if output_data_type == 'same':
		Yₒ = wuml.ensure_data_type(Yₒ, type_name=wtype(Y))
	else:
		Yₒ = wuml.ensure_data_type(Yₒ, type_name=output_data_type)

	return Yₒ

def one_hot_to_label(Yₒ):
	Y = np.argmax(Yₒ, axis=1)
	return Y

def compute_Degree_matrix(M, output_type=None):
	if output_type is None: output_type = wtype(M)

	Mnp = ensure_numpy(M)
	D = np.diag(np.sum(Mnp, axis=0))

	return ensure_data_type(D, type_name=output_type)

