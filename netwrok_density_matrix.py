import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc

#constants
N_dim = 25
t_max= 1000
repetition = 100
sequence_of_nodes = np.zeros(N_dim)
sequence_of_nodes[1] = 1
density_matrix = np.outer(sequence_of_nodes, sequence_of_nodes)



#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim-1):
    Adjacency[i, i+1] = 3
    Adjacency[i+1, i] = 1
Adjacency[N_dim-1, 0] = 3
Adjacency[0, N_dim-1] = 1
""" Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim):
    for j in range(N_dim):
        Adjacency[i, j] = random.random() """

#adjacency normalization
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency[i, j]
    for k in range(N_dim):
        Adjacency[i, k] /= sum
# laplacian 
Laplacian = np.identity(N_dim) - Adjacency

#diagonalization 
Lap_eigenvalue, Lap_eigenvector = np.linalg.eig(Laplacian)
idx = Lap_eigenvalue.argsort()[::]    #sort the eigenvalue and eigenstate
Lap_eigenvalue = Lap_eigenvalue[idx]
Lap_eigenvector = Lap_eigenvector[:,idx]
Lap_eigenvector = np.matrix.transpose(Lap_eigenvector) #the eigenstate were in the column
Lap_eigenvalue = np.diag(Lap_eigenvalue)

#evolution operator
def evolution_operator(t):
    return sc.expm(-t*Laplacian)

#entropy
def entropy(t):
    U = evolution_operator(t)
    density_matrix_t = U @ density_matrix @ U.conj().T
    return -(np.trace(density_matrix_t *sc.logm(density_matrix_t)))

#plot
x = np.linspace(0,t_max,t_max)
y_1= np.zeros(t_max)
for i in range(t_max):
    y_1[i] = entropy(x[i])

y_2 = np.zeros(t_max)
for i in range(t_max):
    y_2[i] = (np.trace(evolution_operator(x[i]) @ density_matrix @ evolution_operator(x[i])))



plt.plot(x, y_1, label='entropy')
plt.plot(x, y_2, label='trace')
plt.xlabel('time')
plt.ylabel('entropy')
plt.show()





