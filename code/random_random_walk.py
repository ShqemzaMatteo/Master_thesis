import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sc
import random
random.seed(None)

N_dim = 5
time = np.linspace(1,100,100)
initial_distribution = np.zeros(N_dim)
initial_distribution[0]=1

#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim):
    Adjacency[i, (i + 1) % N_dim] = 1 + np.random.normal(scale=0.1)
    Adjacency[(i + 1) % N_dim, i] = 1 - np.random.normal(scale=0.1)

for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency[i, j]
    for k in range(N_dim):
        Adjacency[i, k] /= sum
# laplacian 
Laplacian_ring = np.identity(N_dim) - Adjacency

print(Laplacian_ring)

def random_laplacian(Laplacian):
    for i in range(N_dim):
        for j in range(i, N_dim):
            if(Adjacency[i,j] != 0):
                stochastic = np.random.normal(scale=0.1)
                Laplacian[i,j] += stochastic
                Laplacian[j,i] -= stochastic
    return Laplacian

def evolution(t, Laplacian, index):
    distribution = sc.expm(-t*Laplacian) @ initial_distribution
    return distribution[index]

def random_evolution(t, Laplacian, index):
    rand_lapl= random_laplacian(Laplacian)
    distribution = sc.expm(-t*rand_lapl) @ initial_distribution
    return distribution[index]

y_1=np.vectorize(lambda t: evolution(t,Laplacian_ring,0))(time)
y_2=np.vectorize(lambda t: random_evolution(t,Laplacian_ring,0))(time)

plt.plot(time, y_1, label='Ring')
plt.plot(time, y_2, label='Random')
plt.xlabel('time')
plt.ylabel('p(i)')
plt.title('Probability to be in node 0')
plt.xscale('log')
plt.grid()
plt.legend()
plt.show()