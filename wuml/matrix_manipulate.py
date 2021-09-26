
import wuml
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def sort_matrix_rows_by_a_column(X, column_names):
	df = wuml.ensure_DataFrame(X)
	sorted_df = df.sort_values(by=column_names)

	return sorted_df



def one_hot_encoding(Y, output_data_type='same'): 
	'''
		output_data_type : 'same' implies to use the same data type as input
							else use the data type you want, i.e., 'ndarray' , 'Tensor', 'DataFrame'
	'''
	dtype = wuml.data_type(Y)
	Y1 = wuml.ensure_numpy(Y)

	Y1 = np.reshape(Y1,(len(Y1),1))
	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(Y1)

	if output_data_type == 'same':
		Yₒ = wuml.ensure_data_type(Yₒ, type_name=dtype)
	else:
		Yₒ = wuml.ensure_data_type(Yₒ, type_name=output_data_type)

	return Yₒ

def one_hot_to_label(Yₒ):
	Y = np.argmax(Yₒ, axis=1)
	return Y


Y = np.array([0,0,1,1,2,2,3,3])
Yₒ = one_hot_encoding(Y)
Y2 = one_hot_to_label(Yₒ)
