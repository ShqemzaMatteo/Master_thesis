import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sc
import random

#constants
N_dim = 50
time = np.logspace(-1,3,100)
initial_distribution = np.zeros(N_dim)
initial_distribution[0] = 1

#adjacency matrix for a ring
Adjacency = np.zeros((N_dim, N_dim))
for i in range(N_dim):
    Adjacency[i, (i + 1) % N_dim] = 1
    Adjacency[(i + 1) % N_dim, i] = 1

for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency[i, j]
    for k in range(N_dim):
        Adjacency[i, k] /= sum
# laplacian 
Laplacian_ring = np.identity(N_dim) - Adjacency

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
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency_er[i, j]
    for k in range(N_dim):
        Adjacency_er[i, k] /= sum
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
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency_ba[i, j]
    for k in range(N_dim):
        Adjacency_ba[i, k] /= sum
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
Adjacency_ws = ws_adjacency_matrix(N_dim, 6, 0.2)
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency_ws[i, j]
    for k in range(N_dim):
        Adjacency_ws[i, k] /= sum
# laplacian 
Laplacian_ws = np.identity(N_dim) - Adjacency_ws

#entropy
#entropy
def Shannon_entropy(t, laplacian):
    entropy = 0
    distribution = sc.expm(-t*laplacian)@initial_distribution
    distribution /= np.sum(distribution)
    for d in distribution:
        entropy-= d*np.log(d)
    return entropy

#plot
y_1 = np.vectorize(lambda t: Shannon_entropy(t,Laplacian_ring))(time)
y_2 = np.vectorize(lambda t: Shannon_entropy(t,Laplacian_er))(time)
y_3 = np.vectorize(lambda t: Shannon_entropy(t,Laplacian_ba))(time)
y_4 = np.vectorize(lambda t: Shannon_entropy(t,Laplacian_ws))(time)

plt.plot(time, y_1/N_dim, label='Ring')
plt.plot(time, y_2/N_dim, label='E-R')
plt.plot(time, y_3/N_dim, label='B-A')
plt.plot(time, y_4/N_dim, label='W-S')
plt.xlabel('time')
plt.ylabel('S/N')
plt.title('Shannon Entropy for a random graph')
plt.xscale('log')
#plt.ylim((-0.15 , 1.15))
plt.grid()
plt.legend()
plt.show()