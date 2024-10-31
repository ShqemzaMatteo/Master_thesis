import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc
random.seed(None)

#constants
N_dim = 25
t_max= 200
beta = 1

#adjacency matrix for E-R
def er_adjacency_matrix(n, p):
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency[i,j] = 1
                adjacency[j,i] = 1
    return adjacency

#adjacency matrix for a B-A
def ba_adjacency_matrix(n,m):
    adjacency = np.zeros((n,n))
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    degrees = np.sum(adjacency, axis=1)
    for new_node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            # Choose target node based on preferential attachment
            potential_target = np.random.choice(range(new_node), p=degrees[:new_node]/np.sum(degrees[:new_node]))
            targets.add(potential_target)
        # Connect new node to targets
        for target in targets:
            adjacency[new_node, target] = 1
            adjacency[target, new_node] = 1
            degrees[new_node] += 1
            degrees[target] += 1
    return adjacency



Adjacency = er_adjacency_matrix(N_dim, 0.7)
#adjacency normalization
for i in range(N_dim):
    sum = np.sum(Adjacency[i,:])
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
#uniform distribution
probability_vector_1 = np.ones(N_dim)/N_dim
density_matrix_1 = np.outer(probability_vector_1, probability_vector_1)
#random mixed density matrix
sample = 100
density_matrix_2 = np.zeros((N_dim,N_dim))
alphas = np.ones(sample)
dirichlet = np.random.dirichlet(alpha=alphas, size=1)
for number in range(sample):
    mixed_state = np.zeros(N_dim)
    sum = 0
    for i in range(N_dim):
        mixed_state[i] = random.random()
        sum += mixed_state[i]
    mixed_state /= sum
    density_matrix_2 += dirichlet[0, number] * np.outer(mixed_state,mixed_state)
#Boltzmann weight
density_matrix_3 = sc.expm(-beta*Lap_eigenvalue)
density_matrix_3 /= np.trace(density_matrix_3)
#delta-like
probability_vector_4 = np.zeros(N_dim)
probability_vector_4[1] = 1
density_matrix_4 = np.outer(probability_vector_4, probability_vector_4)
#all one density matrix
density_matrix_5 = np.diag(np.ones(N_dim))

#evolution operator
def evolution_operator(t):
    return sc.expm(-t/2*Laplacian)
#entropy
def Von_Neumann(t, density_matrix):
    U = evolution_operator(t)
    density_matrix_t = U @ density_matrix @ U.conj().T
    density_matrix_t /= np.trace(density_matrix_t)
    return -np.trace(density_matrix_t @ sc.logm(density_matrix_t))

#plot
x = np.arange(0,t_max)
y_1 = np.vectorize(lambda t: Von_Neumann(t,density_matrix_1))(x)
y_2 = np.vectorize(lambda t: Von_Neumann(t,density_matrix_2))(x)
y_3 = np.vectorize(lambda t: Von_Neumann(t,density_matrix_3))(x)
y_4 = np.vectorize(lambda t: Von_Neumann(t,density_matrix_4))(x)
#y_5 = np.vectorize(lambda t: Von_Neumann(t,density_matrix_5))(x)

plt.plot(x, y_1, label='Uniform')
plt.plot(x, y_2, label='random mixed')
plt.plot(x, y_3, label='Maxwell')
plt.plot(x, y_4, label='delta-like')
plt.plot(x, y_5, label='mode-uniform')
plt.xlabel('time')
plt.ylabel('entropy')
plt.ylim((-1 , 5))
plt.grid()
plt.legend()
plt.show()