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
sequence_of_nodes[0,:] = 1

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
    particle_per_node = np.zeros(N_dim)
    particle_per_node[1] = 1
    for time in range (t_max - 1):
        for i in range(N_dim):
            if particle_per_node[i]>0.5:
                path_choice = random.random()
                for j in range(N_dim):
                    if path_choice < Adjacency[i, j]:
                        particle_per_node[i] -=1
                        particle_per_node[j] +=1 
                        sequence_of_nodes[time + 1,rep] += j
                        break
                    else:
                        path_choice -= Adjacency[i,j]
                break

probability_matrix = np.zeros((N_dim, t_max))

# histogram
for time in range(t_max):
    frequency, bin_edge, _ = plt.hist(sequence_of_nodes[time,:], bins=N_dim, range =(0,N_dim) , color='blue', edgecolor='black')
    probability_matrix[:, time] = frequency/repetition
    plt.clf()
    """ title = "histogram time" + str(time)
    plt.title(title)
    plt.xlabel('nodes')
    plt.ylabel('Frequency')
    plt.show() """

""" print(probability_matrix[:,t_max-10])
print(Lap_eigenvector @ probability_matrix[:,t_max-10])
print(Lap_eigenvector @ probability_matrix[:,0]) """

diagonalized_probability_matrix = Lap_eigenvector @ probability_matrix
print(diagonalized_probability_matrix[:,400])

""" def expo(x,eigen, P0):
    return P0*np.exp(-x*eigen) 

#plot
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
        if probability_matrix[j,i] != 0:
            Shannon[i] -= probability_matrix[j,i]*np.log(probability_matrix[j,i])

for i in range(t_max):
    for j in range(N_dim):
        if diagonalized_probability_matrix[j,i] != 0:
            Von_Neumann[i] -= abs(diagonalized_probability_matrix[j,i])*np.log(abs(diagonalized_probability_matrix[j,i]))

plt.plot(np.linspace(0,t_max,t_max), Shannon, label='Shannon')
plt.plot(np.linspace(0,t_max,t_max), Von_Neumann, label='Von Neumann')

#density matrix
probability_vector= np.zeros(N_dim)
probability_vector[1] = 1
density_matrix = np.outer(probability_vector, probability_vector)

#evolution operator
def evolution_operator(t):
    return sc.expm(-t/2*Laplacian)

#entropy
def entropy(t):
    U = evolution_operator(t)
    density_matrix_t = U @ density_matrix @ U.conj().T
    return -np.trace(density_matrix_t *sc.logm(density_matrix_t))

#plot
x = np.linspace(0,t_max,t_max)
y= np.zeros(t_max)
for i in range(t_max):
    y[i] = entropy(x[i])


plt.plot(x, y, label='entropy')
plt.xlabel('time')
plt.ylabel('entropy')
plt.legend()
plt.show()