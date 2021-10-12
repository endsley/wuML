
from scipy.integrate import quad
from numpy import linalg as LA
import numpy as np
import wuml


def integrate(foo, x0, x1):
	[result, error] = quad(foo, x0, x1)
	return [result, error]


class eig:
	def __init__(self, M, percentage_to_keep=None, num_eigs_to_keep=None, largest_to_smallest=True,
					return_eigen_values_as_diagonal=True, return_contribution_of_each_vector=False):
		'''
			X = VΛVᵀ
				v0 = wuml.get_np_column(V, 0)
				λ0 = Λ[0]
				λ0*v0 == X.dot(v0)
		'''

		X = wuml.ensure_numpy(M)
	
		if X.shape[0] == X.shape[1]: 		# ensure matrix is square
	
			if np.all(np.abs(X-X.T) < 1e-8):	# matrix is symmetric
				Λ,V = np.linalg.eigh(X)

				if largest_to_smallest:
					Λ = np.flip(Λ)
					V = np.flip(V, axis=1)		# X = VΛVᵀ

				cs = np.cumsum(Λ)/np.sum(Λ)
				if percentage_to_keep is not None:	
					residual_ð = np.sum(cs < percentage_to_keep) + 1
				
				elif num_eigs_to_keep is not None:
					residual_ð = num_eigs_to_keep
				else:
					residual_ð = X.shape[0]

				self.eigen_values = Λ[0:residual_ð]
				self.eigen_vectors = V[:, 0:residual_ð]
				self.cumsum_eigen_values = cs
				self.normalized_eigen_values = Λ/np.sum(Λ)
				self.all_eigen_values = Λ
				self.all_eigen_vectors = V			# each columns is an eigenvector

				if return_eigen_values_as_diagonal:
					self.all_eigen_values = np.diag(Λ)
					self.eigen_values = np.diag(self.eigen_values)

				if return_contribution_of_each_vector:
					self.data_projected_onto_eigvectors = V.T.dot(X)

			else:
				raise ValueError('I have not yet written code to handle non-symmetric matrices for eig')
		else:
			raise ValueError('I have not yet written code to handle non-squared matrices for eig')

	def __str__(self):
		return str(self.eigen_vectors)
