
import sklearn
import wuml

def make_moons(n_samples=100, shuffle=True, noise=0.05, random_state=None):
	(X,Y) = sklearn.datasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)

	data = wuml.ensure_wData(X, column_names=None)
	data.Y = Y
	return data
