
import sklearn.metrics

def rbk_kernel(X, σ, zero_diagonal=False):
	γ = 1.0/(2*σ*σ)
	Kx = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
	if zero_diagonal: np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	return Kx

