import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as sc
random.seed(15)

#constants
N_dim = 50
beta = np.logspace(-2,3,100)

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

#adjacency matrix for WS network
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


#KL divergence
def KL_divergence(b, laplacian_1, laplacian_2):
    density_matrix_1 = sc.expm(-b*laplacian_1)
    density_matrix_1 /= np.trace(density_matrix_1)
    density_matrix_2 = sc.expm(-b*laplacian_2)
    density_matrix_2 /= np.trace(density_matrix_2)
    return np.trace(density_matrix_1 @ sc.logm(density_matrix_1)) -np.trace(density_matrix_1 @ sc.logm(density_matrix_2))

def JS_divergence(b, laplacian_1, laplacian_2):
    density_matrix_1 = sc.expm(-b*laplacian_1)
    density_matrix_1 /= np.trace(density_matrix_1)
    density_matrix_2 = sc.expm(-b*laplacian_2)
    density_matrix_2 /= np.trace(density_matrix_2)
    density_matrix_3 = (density_matrix_1 + density_matrix_2)/2
    return np.trace(density_matrix_1 @ sc.logm(density_matrix_1)) -np.trace(density_matrix_1 @ sc.logm(density_matrix_3))+ np.trace(density_matrix_2 @ sc.logm(density_matrix_2)) -np.trace(density_matrix_2 @ sc.logm(density_matrix_3))
#plot
y_1 = np.vectorize(lambda t: JS_divergence(t,Laplacian_ba, Laplacian_er))(beta)
y_2 = np.vectorize(lambda t: JS_divergence(t,Laplacian_ba, Laplacian_ws))(beta)
y_3 = np.vectorize(lambda t: JS_divergence(t,Laplacian_ws, Laplacian_er))(beta)

plt.plot(beta, y_1,label ='ER - BA')
plt.plot(beta, y_2,label ='ER - WS')
plt.plot(beta, y_3,label ='BA - WS')
plt.xlabel('β')
plt.ylabel('JS')
plt.title('JS divergence between random networks')
plt.xscale('log')
#plt.ylim((-0.15 , 1.15))
plt.grid()
plt.legend()
plt.show()