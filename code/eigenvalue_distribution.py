import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as sc
random.seed(15)

#constants
N_dim = 50
beta = np.logspace(1,3,100)

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
Laplacian = np.identity(N_dim) - Adjacency

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
Adjacency_ws = ws_adjacency_matrix(N_dim, 3, 0.2)
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency_ws[i, j]
    for k in range(N_dim):
        Adjacency_ws[i, k] /= sum
# laplacian 
Laplacian_ws = np.identity(N_dim) - Adjacency_ws

def star_graph_adj(n):
    A = np.zeros((n, n), dtype=int)
    A[0, 1:] = 1
    A[1:, 0] = 1
    return A

Adjacency_star = star_graph_adj(N_dim)
for i in range(N_dim):
    sum = 0
    for j in range(N_dim):
        sum += Adjacency_star[i, j]
    for k in range(N_dim):
        Adjacency_star[i, k] /= sum
# laplacian 
Laplacian_star = np.identity(N_dim) - Adjacency_star

figure, axis = plt.subplots(2, 2)
bin = np.linspace(0,2,9)
print(bin)

eigen_ring = np.linalg.eigvals(Laplacian)
axis[0,0].hist(eigen_ring,bins=bin, label='ring')
axis[0,0].set_xlabel('eigenvalue')
axis[0,0].set_ylabel('count')
axis[0,0].set_title('Ring network')
axis[0,0].grid()

eigen_er = np.linalg.eigvals(Laplacian_er)
axis[0,1].hist(eigen_er,bins=bin, label='ER')
axis[0,1].set_xlabel('eigenvalue')
axis[0,1].set_ylabel('count')
axis[0,1].set_title('E-R network')
axis[0,1].grid()

eigen_ba = np.linalg.eigvals(Laplacian_ba)
axis[1,0].hist(eigen_ba,bins=bin, label='BA')
axis[1,0].set_xlabel('eigenvalue')
axis[1,0].set_ylabel('count')
axis[1,0].set_title('B-A network')
axis[1,0].grid()

eigen_ws = np.linalg.eigvals(Laplacian_ws)
axis[1,1].hist(eigen_ws,bins=bin, label='WS')
axis[1,1].set_xlabel('eigenvalue')
axis[1,1].set_ylabel('count')
axis[1,1].set_title('W-S network')
axis[1,1].grid()

figure.suptitle('Eigenvalues of the Laplacian',fontsize=16)
plt.show()