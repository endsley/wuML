
from sklearn.svm import SVC
import numpy as np

class svm:
	def __init__(self, X, Y, kernel='rbf'):
		if len(Y.shape) == 2: Y = np.squeeze(Y)
		self.SVM = SVC(kernel=kernel)
		self.SVM.fit(X, Y)

	def __call__(self, X):
		return self.SVM.predict(X)
