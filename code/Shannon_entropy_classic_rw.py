import matplotlib.pyplot as plt
import numpy as np
import random

#constants
N_dim = 50
time = np.logspace(-1,2,100)
time_f = time[1:]
time_f = np.append(time_f, 100)
delta_t= time - time_f
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
Adjacency_er = er_adjacency_matrix(N_dim, 0.8)
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
Adjacency_ba = ba_adjacency_matrix(N_dim, 5)
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
def Shannon_entropy(laplacian):
    distribution = initial_distribution
    entropy = np.zeros_like(time)
    for i in range(len(time)):
        distribution += delta_t[i] * (laplacian @ distribution)
        distribution[N_dim-1] = 1- np.sum(distribution[:N_dim-1])
        for d in distribution:
            if(d>0.001):
                entropy[i] -= d*np.log(d)   
    return entropy

#plot
y_1 = Shannon_entropy(Laplacian_ring)
y_2 = Shannon_entropy(Laplacian_er)
y_3 = Shannon_entropy(Laplacian_ba)
y_4 = Shannon_entropy(Laplacian_ws)

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