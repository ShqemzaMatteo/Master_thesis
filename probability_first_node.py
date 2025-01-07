import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
import scipy.linalg as sc
random.seed(None)

#constants
N_dim = 50
time = np.logspace(-2,3,100)
initial_distribution = np.zeros(N_dim)
initial_distribution[0] = 1

def normalization(Adjacency):
    for i in range(N_dim):
        sum = 0
        for j in range(N_dim):
            sum += Adjacency[i, j]
        for k in range(N_dim):
            Adjacency[i, k] /= sum
    return Adjacency

#adjacency matrix for a ring
Adjacency_ring = np.zeros((N_dim, N_dim))
for i in range(N_dim-1):
    Adjacency_ring[i, (i + 1) % N_dim] = 1
    Adjacency_ring[(i + 1) % N_dim, i] = 1

Adjacency_ring= normalization(Adjacency_ring)
# laplacian 
Laplacian_ring = np.identity(N_dim) - Adjacency_ring

# e-r adjacency matrix
def er_adjacency_matrix(n, p):
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency[i,j] = 1
                adjacency[j,i] = 1
    return adjacency
Adjacency_er = er_adjacency_matrix(N_dim, 0.2)
Adjacency_er = normalization(Adjacency_er)
# laplacian 
Laplacian_er = np.identity(N_dim) - Adjacency_er
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
Adjacency_ba = normalization(Adjacency_ba)
    
# laplacian 
Laplacian_ba = np.identity(N_dim) - Adjacency_ba
def ws_adjacency_matrix(n, k, p):
    adjacency= np.zeros((n, n))
    # Step 1: Create a regular ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            # Connect each node to k/2 neighbors on each side (mod n for wrap-around)
            adjacency[i, (i + j) % n] = 1
            adjacency[(i + j) % n, i] = 1
            adjacency[i, (i - j) % n] = 1
            adjacency[(i - j) % n, i] = 1

    # Step 2: Rewire edges with probability p
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if np.random.rand() < p:
                # Remove original connection
                adjacency[i, (i + j) % n] = 0
                adjacency[(i + j) % n, i] = 0
                # Add new edge to a randomly chosen node
                while True:
                    new_connection = np.random.randint(0, n)
                    # Ensure no self-loops or duplicate edges
                    if new_connection != i and adjacency[i, new_connection] == 0:
                        adjacency[i, new_connection] = 1
                        adjacency[new_connection, i] = 1
                        break
    return adjacency
Adjacency_ws = ws_adjacency_matrix(N_dim, 3, 0.2)
Adjacency_ws = normalization(Adjacency_ws)
# laplacian 
Laplacian_ws = np.identity(N_dim) - Adjacency_ws

def evolution(t, Laplacian, index):
    distribution = sc.expm(-t*Laplacian) @ initial_distribution
    return distribution[index]
    

y_1 = np.vectorize(lambda t: evolution(t,Laplacian_ring,0))(time)
y_2 = np.vectorize(lambda t: evolution(t,Laplacian_er,0))(time)
y_3 = np.vectorize(lambda t: evolution(t,Laplacian_ba,0))(time)
y_4 = np.vectorize(lambda t: evolution(t,Laplacian_ws,0))(time)

def check():
    somma = 0
    for i in range(N_dim):
        add = evolution(100,Laplacian_ba,i)
        somma += add
    print(somma)
check()

plt.plot(time, y_1, label='Ring')
plt.plot(time, y_2, label='E-R')
plt.plot(time, y_3, label='B-A')
plt.plot(time, y_4, label='W-S')
plt.xlabel('time')
plt.ylabel('p(i)')
plt.title('Probability to be in node 0')
plt.xscale('log')
plt.grid()
plt.legend()
plt.show()