
import sklearn
import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
from sklearn.datasets import make_classification

def make_moons(n_samples=100, shuffle=True, noise=0.05, random_state=None):
	(X,Y) = sklearn.datasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)

	data = wuml.ensure_wData(X, column_names=None)
	data.Y = Y
	return data

def make_classification_data( n_samples=100, n_features=20, n_informative=2, n_redundant=0, n_repeated=0, 
								n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, 
								hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None):

	X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, 
								n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=weights, flip_y=flip_y, class_sep=class_sep, 
								hypercube=hypercube, shift=shift, scale=scale, shuffle=shuffle, random_state=random_state)


	data = wuml.wData(X_npArray = X, Y_npArray=y, label_type='discrete')
	return data


