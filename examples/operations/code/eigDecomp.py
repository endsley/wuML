#!/usr/bin/env python

import wuml 
import numpy as np

M = wuml.gen_squared_symmetric_matrix(4, distribution='Uniform')
data = wuml.wData(X_npArray=M)

E = wuml.eig(data, num_eigs_to_keep=2, return_contribution_of_each_vector=True)
[Λ, V, Λₐ, Vₐ]= [E.eigen_values, E.eigen_vectors, E.all_eigen_values, E.all_eigen_vectors]
M_back = Vₐ.dot(Λₐ).dot(Vₐ.T)
print('Eigenvalues and Eigenvectors')
wuml.block_two_string_concatenate(Λₐ, Vₐ, spacing='\t', add_titles=['Eigenvalues', 'Eigenvectors'], auto_print=True)

print('Making sure the eigenvectors and values can returns the same matrix.\n\t(M and M_back should be identical)')
wuml.block_two_string_concatenate(M, M_back, spacing='\t', add_titles=['M', 'M from Eigs'], auto_print=True)

wuml.pretty_np_array(E.data_projected_onto_eigvectors, title='Project X onto V : VᵀX', auto_print=True)

print('Study how quickly the eigenvalues grow')
wuml.pretty_np_array(E.cumsum_eigen_values, title='cumsum eigenvalues', auto_print=True)

print('Study the weights of the eigenvalues')
wuml.pretty_np_array(E.normalized_eigen_values, title='Normalized eigenvalues', auto_print=True)

import pdb; pdb.set_trace()
