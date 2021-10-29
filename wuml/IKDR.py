
import sklearn.metrics
from wuml.type_check import *


class IKDR:
	'''
	Interpretable Kernel Dimension Reduction
	'''

	def __init__(self, data, q=2, y=None, y_column_name=None, kernel='rbf', max_rep=8, rank_constrain_ratio=0):
		'''
			rank_constrain_ratio: this constraint lowers the rank of W, default 0 removes the constraint
			q: the number of dimension you want to reduce down to
		'''
		self.X = X = ensure_numpy(data)
		self.Y = Y = ensure_label(data, y=y, y_column_name=y_column_name)
		self.σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))

		Yₒ = wuml.one_hot_encoding(Y)
		Kᵧ = Yₒ.dot(Yₒ.T)
		self.Γ = wuml.double_center(Kᵧ)	# HKᵧH
		self.ζ = rank_constrain_ratio
		self.max_rep = max_rep
		self.conv_threshold = 0.01
		self.q = q

		[self.W, self.λ] = self.update_W(q, W=None)

	def update_W(self, q=None, W=None):
		X, Γ, σ = self.X, self.Γ, self.σ

		Φ = self.gaussian_Φ(X, Γ, σ, W=W)
		return self.eig_solver(Φ, q)


	def eig_solver(self, L, q, mode='smallest'):
		eigenValues,eigenVectors = np.linalg.eigh(L)
	
		if mode == 'smallest':
			U = eigenVectors[:, 0:q]
			U_λ = eigenValues[0:q]
		elif mode == 'largest':
			n2 = len(eigenValues)
			n1 = n2 - q
			U = eigenVectors[:, n1:n2]
			U_λ = eigenValues[n1:n2]
		else:
			raise ValueError('unrecognized mode : ' + str(mode) + ' found.')
		
		return [U, U_λ]



	def gaussian_Φ(self, X, Γ, σ, W=None):
		Ɗ = wuml.compute_Degree_matrix(Γ)
	
		if W is None:
			Φ = X.T.dot(Ɗ - Γ).dot(X)
		else:	
			Kx = wuml.rbk_kernel(X.dot(W), σ)
			Ψ=Γ*Kx
			D_Ψ = wuml.compute_Degree_matrix(Ψ)
			Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
			Φ = self.add_rank_constraint(Φ)

		return Φ

	def add_rank_constraint(self, Φ):		# from the log det rank constraint
		if self.ζ == 0: return Φ
		W, X, ζ = self.W, self.X, self.ζ
	
		A = W.dot(W.T) + 0.001*np.eye(X.shape[1])
		rc = np.linalg.inv(A)
	
		Φ_norm = np.linalg.norm(Φ)
		rc_norm = np.linalg.norm(rc)
		return Φ + (ζ*Φ_norm/rc_norm)*rc

	def fit(self, data, Y):
		Γ, σ, q, λᑊ = self.Γ, self.σ, self.q, self.λ
		X = ensure_numpy(data)

		for i in range(self.max_rep):
			[Ŵ, λᒾ] = self.update_W(q=q, W=self.W)
			Δ = np.linalg.norm(λᑊ- λᒾ)/np.linalg.norm(λᑊ)
			H = wuml.HSIC(X.dot(Ŵ),Y)

			self.W = Ŵ
			λᑊ = λᒾ

			if Δ < self.conv_threshold: break

		reduced_X = X.dot(self.W)
		self.classifier = wuml.classification(reduced_X, y=Y, classifier='SVM')

	def predict(self, data):
		X = ensure_numpy(data)
		reduced_X = X.dot(self.W)

		return self.classifier(reduced_X)
