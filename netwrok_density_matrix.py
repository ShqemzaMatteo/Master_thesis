import matplotlib.pyplot as plt
import numpy as np
import random
import secrets
from scipy.optimize import curve_fit
import scipy.linalg as sc
random_seed = secrets.randbits(32)
random.seed(random_seed)

#constants
N_dim = 25
t_max= 5000
beta = 0.02

#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim-1):
    Adjacency[i, i+1] = 1
    Adjacency[i+1, i] = 1
Adjacency[N_dim-1, 0] = 1
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

#inizial state
#probability_vector = np.ones(N_dim)/N_dim
#density_matrix = np.outer(probability_vector, probability_vector)
""" density_matrix = np.zeros((N_dim,N_dim))
for number in range(100):
    mixed_state = np.zeros(N_dim)
    sum = 0
    for i in range(N_dim):
        mixed_state[i] = random.random()
        sum += mixed_state[i]
    mixed_state /= sum
    density_matrix += np.outer(mixed_state,mixed_state) """
    
density_matrix = sc.expm(-beta*Laplacian)

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

""" y_2 = np.zeros(t_max)
for i in range(t_max):
    y_2[i] = (np.trace(evolution_operator(x[i]) @ density_matrix @ evolution_operator(x[i]))) """

plt.plot(x, y_1, label='entropy')
#plt.plot(x, y_2, label='trace')
plt.xlabel('time')
plt.ylabel('entropy')
plt.show()





