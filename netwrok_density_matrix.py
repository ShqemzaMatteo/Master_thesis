import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc
random.seed(None)

#constants
N_dim = 35
t_max= 500
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

#initial state
probability_vector_1 = np.ones(N_dim)/N_dim
density_matrix_1 = np.outer(probability_vector_1, probability_vector_1)
sample = 10
density_matrix_2 = np.zeros((N_dim,N_dim))
alphas = np.ones(sample)
dirichlet = np.random.dirichlet(alpha=alphas, size=1)
print(dirichlet)
for number in range(sample):
    mixed_state = np.zeros(N_dim)
    sum = 0
    for i in range(N_dim):
        mixed_state[i] = random.random()
        sum += mixed_state[i]
    mixed_state /= sum
    density_matrix_2 += dirichlet[0, number] * np.outer(mixed_state,mixed_state)
density_matrix_3 = sc.expm(-beta*Lap_eigenvalue)
density_matrix_3 /= np.trace(density_matrix_3)
probability_vector_4 = np.zeros(N_dim)
probability_vector_4[1] = 1
density_matrix_4 = np.outer(probability_vector_4, probability_vector_4)

#evolution operator
def evolution_operator(t):
    return sc.expm(-t/2*Laplacian)
#entropy
def entropy(t, density_matrix):
    U = evolution_operator(t)
    density_matrix_t = U @ density_matrix @ U.conj().T
    density_matrix_t /= np.trace(density_matrix_t)
    return -np.trace(density_matrix_t @ sc.logm(density_matrix_t))

#plot
x = np.arange(0,t_max)
y_1 = np.vectorize(lambda t: entropy(t,density_matrix_1))(x)
y_2 = np.vectorize(lambda t: entropy(t,density_matrix_2))(x)
y_3 = np.vectorize(lambda t: entropy(t,density_matrix_3))(x)
y_4 = np.vectorize(lambda t: entropy(t,density_matrix_4))(x)
 
""" y_2 = np.zeros(t_max)
for i in range(t_max):
    y_2[i] = (np.trace(evolution_operator(x[i]) @ density_matrix @ evolution_operator(x[i]))) """

plt.plot(x, y_1, label='Uniform')
plt.plot(x, y_2, label='random mixed')
plt.plot(x, y_3, label='Maxwell')
plt.plot(x, y_4, label='delta-like')
#plt.plot(x, y_2, label='trace')
plt.xlabel('time')
plt.ylabel('entropy')
plt.ylim((-1 , 5))
plt.grid()
plt.legend()
plt.show()





