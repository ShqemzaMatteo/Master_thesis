import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc

#constants
N_dim = 25
Temp = 20
beta = 1/Temp
t_max= 500
repetition = 100
sequence_of_nodes = np.zeros((t_max, repetition))
"""sequence_of_nodes[0,:] = 1
 probability_vector = np.zeros(N_dim)
probability_vector[1] = 1
density_matrix = np.outer(probability_vector, probability_vector) """
sequence_of_nodes[0,:] = np.random.randint(0,N_dim,size=repetition)
probability_vector_1 = np.ones(N_dim)/N_dim
density_matrix = np.outer(probability_vector_1, probability_vector_1)

#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim-1):
    Adjacency[i, i+1] = 1
    Adjacency[i+1, i] = 1
Adjacency[N_dim-1, 0] = 1
Adjacency[0, N_dim-1] = 1

#adjacency normalization
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency[i, j]
    for k in range(N_dim):
        Adjacency[i, k] /= sum

# laplacian 
Laplacian = np.identity(N_dim) -Adjacency

#diagonalization 
Lap_eigenvalue, Lap_eigenvector = np.linalg.eig(Laplacian)
idx = Lap_eigenvalue.argsort()[::]    #sort the eigenvalue and eigenstate
Lap_eigenvalue = Lap_eigenvalue[idx]
Lap_eigenvector = Lap_eigenvector[:,idx]
Lap_eigenvector = np.matrix.transpose(Lap_eigenvector) #the eigenstate were in the column
Lap_eigenvalue = np.diag(Lap_eigenvalue)

#dynamics
for rep in range(repetition):
    for time in range (t_max - 1):
            index = int(sequence_of_nodes[time, rep])
            path_choice = random.random()
            for j in range(N_dim):
                if path_choice < Adjacency[index, j]: 
                    sequence_of_nodes[time + 1, rep] += j
                    break
                else:
                    path_choice -= Adjacency[index, j]

probability_matrix = np.zeros((N_dim, t_max))
# histogram
for time in range(t_max):
    frequency = plt.hist(sequence_of_nodes[time,:], bins=N_dim, range =(0,N_dim) , color='blue', edgecolor='black')[0]
    probability_matrix[:, time] = frequency/repetition
    plt.clf()

diagonalized_probability_matrix = Lap_eigenvector @ probability_matrix
diagonalized_probability_matrix /=  diagonalized_probability_matrix.sum()

""" #plot
def expo(x,eigen, P0):
    return P0*np.exp(-x*eigen) 

cmap=plt.get_cmap('gist_rainbow')
for i in range(4):
    plt.plot(np.linspace(0,t_max,t_max), diagonalized_probability_matrix[i, :], color=cmap(i/N_dim),label=f'node {i}')
    popt, pcov = curve_fit(expo, np.linspace(0,t_max,t_max), diagonalized_probability_matrix[i, :], (Lap_eigenvalue[i,i], diagonalized_probability_matrix[i,0]))
    print(Lap_eigenvalue[i,i], popt)
    plt.plot(np.linspace(0,t_max,t_max), expo(np.linspace(0,t_max,t_max), popt[0], popt[1]), color=cmap(i/N_dim))
plt.ylim([-0.5,0.5])
plt.grid()
plt.legend(loc='upper right')
plt.show() """

#entropy
Shannon = np.zeros(t_max)
Von_Neumann = np.zeros(t_max)
for i in range(t_max):
    for j in range(N_dim):
        if probability_matrix[j,i] > 0.001:
            Shannon[i] -= probability_matrix[j,i]*np.log(probability_matrix[j,i])
        if diagonalized_probability_matrix[j,i] != 0:
            Von_Neumann[i] -= diagonalized_probability_matrix[j,i]*np.log(abs(diagonalized_probability_matrix[j,i]))

plt.plot(np.arange(0,t_max), Shannon, label='Shannon exp')
plt.plot(np.arange(0,t_max), Von_Neumann, label='Von Neumann exp')

#evolution operator
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
x = np.arange(0, t_max)
y = np.vectorize(lambda t: entropy(t, density_matrix))(x)

plt.plot(x, y, label='Von Neumann predicted')
plt.xlabel('time')
plt.ylabel('entropy')
plt.title('Entropies for a uniform distribution')
plt.legend()
plt.grid()
plt.ylim((-1 , 5))
plt.show() 