import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc
random.seed(None)

#constants
N_dim = 50
beta = np.arange(0,36)

#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim-1):
    Adjacency[i, i+1] = 1
    Adjacency[i+1, i] = 1
Adjacency[N_dim-1, 0] = 1
Adjacency[0, N_dim-1] = 1
# e-r adjacency matrix
def er_adjacency_matrix(n, p):
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency[i,j] = 1
                adjacency[j,i] = 1
    return adjacency
Adjacency_er = er_adjacency_matrix(N_dim, 0.7)
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
Adjacency_ba = ba_adjacency_matrix(N_dim, 3)


def Laplacian(adjacency):
    #adjacency normalization
    for i in range(N_dim):
        sum = 0
        for j in range(N_dim):
            sum += adjacency[i, j]
        for k in range(N_dim):
            adjacency[i, k] /= sum
    # laplacian 
    Laplacian = np.identity(N_dim) - adjacency
    #diagonalization 
    Lap_eigenvalue, Lap_eigenvector = np.linalg.eig(Laplacian)
    idx = Lap_eigenvalue.argsort()[::]    #sort the eigenvalue and eigenstate
    Lap_eigenvalue = Lap_eigenvalue[idx]
    Lap_eigenvector = Lap_eigenvector[:,idx]
    Lap_eigenvector = np.matrix.transpose(Lap_eigenvector) #the eigenstate were in the column
    Lap_eigenvalue = np.diag(Lap_eigenvalue)
    return Laplacian, Lap_eigenvalue, Lap_eigenvector

#entropy
def Von_Neumann(b, adjacency):
    Laplacian, _, _ = Laplacian(adjacency)
    density_matrix = sc.expm(-b*Laplacian)
    density_matrix /= np.trace(density_matrix)
    return -np.trace(density_matrix @ sc.logm(density_matrix))

#plot
y_1 = np.vectorize(lambda t: Von_Neumann(t,Adjacency))(beta)
y_2 = np.vectorize(lambda t: Von_Neumann(t,Adjacency_er))(beta)
y_3 = np.vectorize(lambda t: Von_Neumann(t,Adjacency_ba))(beta)

plt.plot(beta, y_1, label='Boltzmann_weight')
plt.plot(beta, y_2, label='Boltzmann_weight')
plt.plot(beta, y_3, label='Boltzmann_weight')
plt.xlabel('time')
plt.ylabel('entropy')
plt.ylim((-1 , 5))
plt.grid()
plt.legend()
plt.show()