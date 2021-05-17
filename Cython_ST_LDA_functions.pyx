#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:46:29 2021

@author: juliafoyer
"""
import random
import numpy as np
import scanpy as sc
import anndata as an
import scipy.stats as st
from scipy.spatial import KDTree
import math
from numba import njit
from numba import jit
from typing import *

import matplotlib.pyplot as plt
from matplotlib import rcParams

DEBUG = False

def prepare_data(adata : an.AnnData, select_hvg : bool = True)-> an.AnnData:
    adata.var_names_make_unique()
    X = np.array(adata.X.todense())
    adata.X = X
    if select_hvg:
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes = 1000)
#        adata = adata[:,adata.var.highly_variable_genes.values]
        adata = adata[:,adata.var.highly_variable.values]
        adata.X = (np.exp(adata.X) -1).round(0)
    return adata

def subsample(adata):
    n_spots,n_genes = adata.X.shape
#    N_s = adata.X.sum(axis=1)
    for spot in range(n_spots):
  #      print(N_s[spot])
#        p_s = X[spot,:] / N_s[spot] # one per gene, i.e, 1000
        adata.X[spot] = np.ceil(adata.X[spot] / 10)
 #       p_s = np.asarray(p_s).astype(np.float64)
 #       print(p_s.sum())
#        a = int(np.ceil(0.1*N_s[spot]))
#        print(a)
#        new_x_s = np.random.choice(a, p_s)
#        X[spot,:] = new_x_s
    return adata  
    
    
#    p_s = np.zeros(n_spots, n_genes)
#    p_s = X(axis=1) / X(axis=1).sum()
#    print(p_s)
#    p_s = x_s  / x_s.sum() # p-values for spot
#    new_x_s = np.random.multiomial(np.ceil(0.1*N_s),p_s)

def assign_random(adata, K):
    """ Assigns a random factor in range(K) to each
    UMI.

    Parameters:
    ----------
    adata: an.AnnData
        anndata object to study
    K: int
        number of topics
    
    Returns:
    ----------
    A list of lists (one for each spot), containing
    a random factor for each UMI.
    """
    umi_factor_list = []
    all_n_umis = adata.X.sum(axis=1)
    for n_UMIs in all_n_umis:
        factors = np.random.randint(low = 0,
                                    high = K,
                                    size = int(n_UMIs))
        umi_factor_list.append(factors)
    return umi_factor_list

def get_ids_dt_wt(adata: an.AnnData,
                  umi_factors: List[np.ndarray],
                  K: int,
                  alpha = 0.1,
                  beta = 0.1)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ add short description here
    
    Parameters:
    ----------
    adata : an.AnnData
        anndata object to study
    umi_factors. List[np.ndarray]
        List of arrays. Element j in list i represents
        topic that word j belongs to in spot i
    K: int
        number of topics

    Returns:
    --------
    
    """
    
    n_spots, n_genes = adata.X.shape
    X = adata.X
    ids = []
    dt = np.zeros((n_spots, K)) + alpha
    wt = np.zeros((K, n_genes)) + beta
    
    for spot in range(n_spots):
        ids_spot = []
        spot_list = umi_factors[spot].tolist()
        start = 0
        end = 0
        for gene in range(n_genes):
            n_umis = int(X[spot, gene])
            ids_spot += [gene] * n_umis
            end += n_umis
            for factor in range(K):
                wt[factor, gene] += spot_list[start: end].count(factor)
            start = end
        ids.append(ids_spot)
        for factor in range(K):
            dt[spot, factor] += spot_list.count(factor)
    return ids, dt, wt

def get_nz(dt):
    nz = dt.sum(axis=0)
    return nz

def get_theta(dt):
    """ add short description here
    
    Parameters:
    ----------
    """
    theta = dt / dt.sum(axis=1, keepdims=True)
    return theta

def gibbsSampling(umi_factors,
                  ids,
                  dt,
                  wt,
                  nz,
                  iterations):
    
    for it in range(iterations):
        for d, doc in enumerate(ids):
            for index, w in enumerate(doc):
                z = umi_factors[d][index]
                dt[d, z] -= 1
                wt[z, w] -= 1
                nz[z] -= 1
                pz = np.divide(np.multiply(dt[d, :], wt[:, w]), nz)
                z = np.random.multinomial(1, (pz / pz.sum())).argmax()
                umi_factors[d][index] = z 
                dt[d, z] += 1
                wt[z, w] += 1
                nz[z] += 1
                
def draw_new_factor(umi_factors,
                    dt,
                    wt,
                    nz,
                    d,
                    index,
                    w):
    z = umi_factors[d][index]
    dt[d, z] -= 1
    wt[z, w] -= 1
    nz[z] -= 1
    pz = np.divide(np.multiply(dt[d, :], wt[:, w]), nz)
    z = np.random.multinomial(1, (pz / pz.sum())).argmax()
    umi_factors[d][index] = z 
    dt[d, z] += 1
    wt[z, w] += 1
    nz[z] += 1

def plot_theta(adata, dt):
    theta = get_theta(dt)
    side_length = adata.uns["info"]["side_length"]
    plt.imshow(theta.reshape(side_length,side_length,3))
    plt.title("Inferred theta values (in RGB space) across tissue")
    plt.show()

        
def build_nbrhd(adata: an.AnnData,
            n_neighbours: int)->Tuple[List[np.ndarray],List[np.ndarray]]:
    """ add short description here
    
    Parameters:
    ----------
    """
    kd = KDTree(adata.obsm["spatial"])
    dist,indx = kd.query(adata.obsm["spatial"], k = n_neighbours) # why not +1?
    no_nbr = np.isinf(dist)
    dist[no_nbr] = 0
    dist[~no_nbr] = 1
    indx[no_nbr] = -1
    return dist, indx

def findKNN(adata: an.AnnData,
            n_neighbours: int,
            max_distance: float)->Tuple[List[np.ndarray],List[np.ndarray]]:
    """ add short description here
    
    Parameters:
    ----------
    """
    kd = KDTree(adata.obsm["spatial"])
    dist,indx = kd.query(adata.obsm["spatial"],
                         k = n_neighbours + 1,
                         distance_upper_bound = max_distance, #I added this
                        )
    return dist, indx

def remove_false_neighbours(dist, indx, eps = 300):
    """ add short description here
    
    Parameters:
    ----------
    """
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

def get_E(indx_sel):
    """ add short description here
    
    Parameters:
    ----------
    """
    edges = 0
    for i in indx_sel:
        edges += len(i) - 1
    edges = edges / 2
    return edges

@njit(parallel=False)
def bhattacharyya_distance(p: np.ndarray,
                           q: np.ndarray,
                           X: np.ndarray)->float:
    """ add short description here
    
    Parameters:
    ----------
    p: np.ndarray
        a probability distribution, e.g., a theta vector
    q: np.ndarray'
        another probability distribution, e.g., a theta vector
    X: np.ndarray
        the domain of the distributions
    
    Returns:
    --------
    the bhattacharyya distance (float) between the two
    distributions p and q.
    
    """
    BC = 0
    for x in range(X):
        BC += math.sqrt(p[x]*q[x])
    DB = - math.log(BC)
    return DB

def distance_fun(p,q):
    #Bhattacharyya distance
    return - np.log(np.sqrt(p*q).sum())

def log_potential(edge_influence, current_theta, neighbour_theta):
#    return edge_influence * bhattacharyya_distance(current_theta, neighbour_theta, K)
    return edge_influence * distance_fun(current_theta, neighbour_theta)

def first_edge_influence(lambda0: float,
                         theta: np.ndarray,
                         indx_sel: List[np.ndarray],
                         K: int)->np.ndarray:
    
    """ add short description here
    
    Parameters:
    ----------
    lambda0: float
    theta: np.ndarray
    indx_sel: List[np.ndarray]
    K: int
    
    Returns:
    ----------
    np.ndarray
    """
    
    edge_influence = []
    n_spots = len(indx_sel)
    for spot in range(n_spots):
        p = theta[0]
        influences = []
        neighbours = indx_sel[spot][1:]
        for neighbour in neighbours:
            q = theta[neighbour]
            influences.append(np.random.exponential(lambda0 + bhattacharyya_distance(p, q, K)))
        edge_influence.append(influences)
    return np.array(edge_influence, dtype=object)

def sample_edge_influence(edge_influence: np.ndarray,
                          lambda_parameter: float,
                          theta: np.ndarray,
                          indx_sel: List[np.ndarray],
                          K: int)->None:
    
    n_spots = len(indx_sel)
    X = K
    for spot in range(n_spots): # enumerate instead of using index?
        p = theta[spot]
#        index = 1
        n_neighbours = len(indx_sel[spot][1:]) # I can calculate this once
        for index in range(n_neighbours):
            neighbour = indx_sel[spot][index+1]
            q = theta[neighbour]
            edge_influence[spot][index] = (np.random.exponential(lambda_parameter + bhattacharyya_distance(p, q, X)))

def sample_edge_influence2(edge_influence: np.ndarray,
                          lambda_parameter: float,
                          theta: np.ndarray,
                          indx_sel: List[np.ndarray],
                          K: int)->None:
    
    n_spots = len(indx_sel)
    X = K
    for spot in range(n_spots): # enumerate instead of using index?
        n_neighbours = len(indx_sel[spot][1:]) # I can calculate this once
        for index in range(n_neighbours):
            neighbour = indx_sel[spot][index+1]
            edge_influence[spot][index] = 1
        
            
def sample_lambda(E: int,
                  edge_influence: float,
                  lambda_a = 0.01,
                  lambda_b = 0.01):
    
    """ add short description here
    
    Parameters:
    ----------
    """
    
    sum_edge_influences = 0
    for edge in edge_influence:
        sum_edge_influences += sum(edge)
    shape = lambda_a + E
    rate = 1 / (lambda_b + sum_edge_influences)
    lambda_parameter = np.random.gamma(shape, rate, 1)
    return lambda_parameter

def metropolis_hastings(spot,
                        theta,
                        dt,
                        indx_sel,
                        alpha_orig,
                        edge_influence,
                        n_iter = 20):
    
    n_topics = theta.shape[1]
    accepted = 0
    
    def log_target_dist(theta_eval):
        l_prob = 0
        for k,nbr in enumerate(indx_sel[spot][1::]): # going through all neighbours, k=counter
            l_prob -=  edge_influence[spot][k]*\
            bhattacharyya_distance(theta_eval,theta[nbr],n_topics)
        return l_prob
    
    proposal_dist = st.dirichlet(alpha_orig + dt[spot,:])
    
    old_theta = proposal_dist.rvs()[0]
    old_log_proposal = proposal_dist.logpdf(old_theta)
    old_log_target = log_target_dist(old_theta)
            
    for it in range(n_iter):
        new_theta = proposal_dist.rvs()[0]
        new_log_proposal = proposal_dist.logpdf(new_theta)
        new_log_target = log_target_dist(new_theta)
        
        log_u = new_log_target + old_log_proposal - old_log_target - new_log_proposal
        u = np.exp(log_u) # I get an overflow warning here, but I hope it's not a problem.
        a = min(u,1) # If overflow I hope it rounds, and it will always be bigger than 1.
        if np.random.random() < a:
            old_theta = new_theta
            old_log_proposal = new_log_proposal
            old_log_target = new_log_target
            
            accepted += 1
        
    if DEBUG: print("Fraction of accepted proposals : {}".format(accepted / n_iter))
            
    return old_theta

def sample_theta(spot, theta, dt, indx, dist, edge_influence, K, n_iter = 10):
        
    proposal_dist = st.dirichlet(dt[spot,:])
        
    theta_mat = np.zeros((n_iter+1, K))
        
    theta_mat[0,:] = theta[spot,:]
    old_log_proposal = proposal_dist.logpdf(theta_mat[0,:])
    old_log_target = log_potential(edge_influence, theta_mat[0,:], theta[indx[spot,:]])
    old_log_target *= dist[spot,:] # Keep only those that are real neighbours. * 0 cancels false neighbours.
    old_log_target = old_log_target.sum()
    
    for it in range(1,n_iter+1):
        new_theta = proposal_dist.rvs()[0] # what is the 0? The first topic?
        new_log_proposal = proposal_dist.logpdf(new_theta)
        new_log_target = log_potential(edge_influence, new_theta, theta[indx[spot,:]])
        new_log_target *= dist[spot,:]
        new_log_target = new_log_target.sum()
                    
        log_u = new_log_target + old_log_proposal - old_log_target - new_log_proposal
        u = np.exp(log_u) 
        a = min(u,1)
        
        if np.random.random() < a:
            theta_mat[it,:] = new_theta
            old_log_target = new_log_target
            old_log_proposal = new_log_proposal
        else:
            theta_mat[it,:] = theta_mat[it-1,:]
                    
        theta[spot,:] = theta_mat.mean(axis=0)
#        theta[spot,:] = theta_mat[6:-1, :].mean(axis=0)
