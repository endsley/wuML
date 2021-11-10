
import numpy as np
import wuml

class feature_selection: 
	def __init__(self, data, Y=None, method='HSIC filter'):
		X = wuml.ensure_wData(data)

		if method == 'HSIC filter':
			self.hsic_filter(X,Y)

	def hsic_filter(self, X, Y):
		d = X.shape[1]
		x = X.X
		col_ids = np.arange(d).tolist()

		highest_hsic_column = None
		highest_hsic = 0
		if Y is None:
			for i in col_ids:
				n = col_ids[:i] + col_ids[i+1:]
				z = x[:, n]

				#H = wuml.HSIC(z,z, sigma_type='mpd', normalize_hsic=True)

				H = wuml.HSIC(z,x, sigma_type='opt', normalize_hsic=True)
				if highest_hsic < H:
					highest_hsic = H
					highest_hsic_column = col_ids[i]
				import pdb; pdb.set_trace()

		import pdb; pdb.set_trace()

#			C = X.columns.to_list()
#			for i in C:
#				z = np.copy(x)
#
#				Xc = X.get_columns(i)
#				H = wuml.HSIC(Xc,X, sigma_type='mpd', normalize_hsic=True)
#				import pdb; pdb.set_trace()
