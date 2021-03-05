# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

# import modules
import random
import numpy as np
import scanpy as sc
import anndata as an
import scipy.stats as st
from scipy.spatial import KDTree

# Load/create data
def LoadData(path):
    adata = sc.read_visium(path)
    adata.var_names_make_unique()
    return adata

# Find nearest neighbours
def findKNN(adata, K):
    kd = KDTree(adata.obsm["spatial"])
    dist,indx = kd.query(adata.obsm["spatial"],k=K)
    return dist, indx

# Find nearest neighbours
def findKNN_ub(adata, K):
    kd = KDTree(adata.obsm["spatial"])
    dist,indx = kd.query(adata.obsm["spatial"],k=K, distance_upper_bound = 300)
    return dist, indx

def remove_false_neighbours(dist, indx, eps = 300):
    dist_sel = []
    indx_sel = []
    for i in range(len(dist)):
        dist_sel.append([])
        indx_sel.append([])
        for j in range(len(dist[i])):
            if dist[i][j] < eps:
                dist_sel[i].append(dist[i][j])
                indx_sel[i].append(indx[i][j])
        dist_sel[i] = np.array(dist_sel[i])
        indx_sel[i] = np.array(indx_sel[i])
    return np.array(dist_sel, dtype=object), np.array(indx_sel, dtype=object)

def KNN_indx_to_barcode(indx):
    KNN_ids = []
    index = 0
    for i in indx:
        KNN_ids.append([])
        for j in i:
            KNN_ids[index].append(adata.obs_names[j])
        KNN_ids[index] = np.array(KNN_ids[index])
        index += 1
    return np.array(KNN_ids, dtype=object)

def A_prob(neighbours, theta0, theta1): # This is not done...
    a = 1
    for spot in neighbours[1:]:
        edge_influence = 2 # Arbitrarily picked for now
        distance = # I need to calculate distances between theta0 and theta1
        potential = st.expon(edge_influence * distance)
        a *= potential
    return a

def metr_hast(adata, K):
    # Prepare proposal distribution    
    dist_g = lambda x : st.dirichlet(x)
    
    # Prepare acceptance probability
    a_prob = lambda neighbours, 
    
    # Initial guess
    alpha = 0.1 * np.ones(K)
    theta0 = dist_g2(alpha)
    print(theta0)  # Why does it look like this when I print?
    
    # Metropolis-Hastings sampling
    spots = adata.n_obs
    theta = theta0
    for spot in range(spots):
        for it in range(10): # "Run 10 metropolis steps per document"
            theta1 = dist_g(theta0)
            a = 
            a = min(1, 0.5) # 0.5 should be a
            if np.random.random < a:
                theta0 = theta1
        theta = np.vstack ((theta, theta0))

# Other parameters - honestly I can't remeber anymore why I started doing this. Maybe because they will be needed later for Gibbs sampling.
def Gibbs():
    lambda_a = 0.01
    lambda_b = 0.01
    E = 3798*5/2
    edge_influence = 2 
    shape = lambda_a + E
    scale = lambda_b + edge_influence*E
    lambda_parameter = np.random.gamma(shape, scale, 1)        

# Main
path ="/Users/juliafoyer/Documents/Skolarbete/Masters_thesis/human_breast_cancer_ST_data"
K = 10
adata = LoadData(path)
dist, indx = findKNN(adata, 7)
print("dist[4]:   ", dist[4])
dist_ub, indx_ub = findKNN_ub(adata, 7)
print("dist_ub[4]:   ", dist_ub[4])
dist_sel, indx_sel = remove_false_neighbours(dist, indx)
KNN_ids = KNN_indx_to_barcode(indx_sel)
print("dist_sel[4]:   ", dist_sel[4])
print("indx[4]:   ", indx[4])
print("indx_sel[4]:   ", indx_sel[4])
print("KNN_ids[4]:   ", KNN_ids[4])
print(dist.shape)
print(dist_ub.shape)
print(dist_sel.shape) # Is this because length of rows now differ?
theta = metr_hast(adata, K)
