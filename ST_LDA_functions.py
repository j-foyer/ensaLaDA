#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:46:29 2021

@author: juliafoyer
"""
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as an
import scipy.stats as st
from scipy.spatial import KDTree
import math
from numba import njit
from numba import jit
from typing import *
from itertools import accumulate
from functools import reduce
import os
import re
import datetime
from scipy.spatial.distance import cdist
import lda
import lda.datasets

import matplotlib.pyplot as plt
from matplotlib import rcParams

DEBUG = False

def prepare_data(adata : an.AnnData, todense : bool = True, select_hvg : bool = True)-> an.AnnData:
    adata.var_names_make_unique()
    if todense:
        X = np.array(adata.X.todense())
        adata.X = X
    if select_hvg:
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes = 1000)
        adata = adata[:,adata.var.highly_variable.values]
        adata.X = (np.exp(adata.X) -1).round(0)
    return adata

def create_object(counts_path, metrics_path):
    obs = pd.read_csv(counts_path, delimiter='\t')
    metrics = pd.read_csv(metrics_path, delimiter='\t')
    xy = obs['Unnamed: 0'].tolist()
    metrics["xy"] = metrics["x"].astype(str) + "x" + metrics["y"].astype(str)
    metrics = metrics.set_index('xy')
    metrics = metrics.reindex(index=obs['Unnamed: 0'])
    metrics = metrics.reset_index()
    pix_xy = metrics.loc[:, ['pixel_x', 'pixel_y']]
    spatial = pix_xy.values
    X = obs
    del X['Unnamed: 0']  
    adata = an.AnnData(X)
    adata.obsm['spatial'] = spatial
    return adata

def subsample(adata):
    n_spots, n_genes = adata.X.shape
    N_s = adata.X.sum(axis=1).astype(np.float64)
    for spot in range(n_spots):
        p_s = adata.X[spot,:].astype(np.float64) / N_s[spot]
        p_s = p_s.astype(np.float64)
        n = int(np.ceil(0.1*N_s[spot]))
        new_x_s = np.random.multinomial(n,p_s)
        adata.X[spot,:] = new_x_s
    return adata  

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

def assign_less_random(adata, K):
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
        factors = []
        for i in range(int(n_UMIs)):
            z = i % K
            factors.append(z)
        umi_factor_list.append(np.array(factors))
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

def get_nz(wt, beta, n_genes):
    nz = wt.sum(axis=1) - n_genes*beta
    return nz

def get_theta(dt):
    """ add short description here
    
    Parameters:
    ----------
    """
    theta = dt / dt.sum(axis=1, keepdims=True)
    return theta

def get_phi(wt):
    phi = wt/wt.sum(axis=1, keepdims=True)
    return phi
    
@njit(parallel=False)
def draw_new_factor_vanilla(sub_list,
                    dt,
                    wt,
                    nz,
                    d,
                    index,
                    w):
    
    z = sub_list[index]
    dt[d, z] -=  1
    wt[z, w] -= 1
    nz[z] -= 1
    pz = np.divide(np.multiply(dt[d, :], wt[:, w]), nz) # beta*words?
    z = np.random.multinomial(1, (pz / pz.sum())).argmax()
    sub_list[index] = z 
    dt[d, z] += 1
    wt[z, w] += 1
    nz[z] += 1

@njit(parallel=False)
def multinomial(N: int, pvals: Union[np.ndarray, list]):
    """Multinomial sampling function
    Parameters
    ----------
    N : int
        Number of draws
    pvals : Union[np.ndarray,list]
        probability vector
    Returns:
    -------
    Numpy array representing a
    sample from Mult(n,p)
    """
    
    n = len(pvals)
    res = [0] * n
    pv = np.zeros(n)    
    pv[0] = pvals[0]
         
    for i in range(1,n):
        pv[i] += pv[i-1] + pvals[i]
               
    for i in range(N):
        r = random.random() * pv[-1]
        for j in range(n):
            if pv[j] >= r:
                res[j] += 1
                break
            
    return np.array(res)

@njit(parallel=False)
def draw_new_factor_vanilla2(sub_list,
                    dt,
                    wt,
                    nz,
                    d,
                    index,
                    w,
                    n_genes,
                    beta):    
    z = sub_list[index]
    dt[d, z] -=  1
    wt[z, w] -= 1
    nz[z] -= 1
    pz = np.divide(np.multiply(dt[d, :], wt[:, w]), nz) # beta*words?
 #   pz = np.divide(np.multiply(dt[d, :], wt[:, w]), (nz + beta * n_genes))
    pz = pz / pz.sum()
    z = multinomial(1, pz).argmax()
    sub_list[index] = z 
    dt[d, z] += 1
    wt[z, w] += 1
    nz[z] += 1
    
    
    
@njit(parallel=False)
def draw_new_factor(sub_list,
                    dt,
                    wt,
                    nz,
                    theta,
                    d,
                    index,
                    w, 
                    beta,
                    n_genes):    
    z = sub_list[index]
    dt[d, z] -=  1
    wt[z, w] -= 1
    nz[z] -= 1
 #   p2 = np.divide(wt[:,w], nz)
#    pz = np.multiply(theta[d,:], wt[:,w])
#    pz = pz / wt[:,w].sum()
#    p2 = wt[:,w] / wt[:,w].sum()
    p2 = wt[:,w] / (nz + beta * n_genes)
    pz = theta[d,:] * p2
    pz = pz / pz.sum()
    z = multinomial(1, pz).argmax()
    sub_list[index] = z 
    dt[d, z] += 1
    wt[z, w] += 1
    nz[z] += 1
    
    return sub_list
        
def build_nbrhd(adata: an.AnnData,
            n_neighbours: int)->Tuple[List[np.ndarray],List[np.ndarray]]:
    """ add short description here
    
    Parameters:
    ----------
    """
    kd = KDTree(adata.obsm["spatial"])
    dist,indx = kd.query(adata.obsm["spatial"], k = n_neighbours)
    no_nbr = np.isinf(dist)
    dist[no_nbr] = 0
    dist[~no_nbr] = 1
    indx[no_nbr] = -1
    return dist, indx

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

@njit(parallel=False)
def distance_fun(p,q):
    #Bhattacharyya distance
    return - np.log(np.sqrt(p*q).sum())

@njit(parallel=False)
def log_potential(edge_influence, current_theta, neighbour_theta):
    return -edge_influence * distance_fun(current_theta, neighbour_theta)

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

@njit(parallel=False)
def log_pdf_dirichlet_old(x,alpha):
    l_beta_top = 0
    l_beta_bot = 0
    frac = 0
    for _a,_x in zip(alpha,x):
        l_beta_top += math.log(math.gamma(_a))
        l_beta_bot += _a
        frac += (_a - 1)*math.log(_x)
    l_beta_bot = np.log(math.gamma(l_beta_bot))
    return frac + l_beta_bot - l_beta_top

@njit(parallel=False)
def dirichlet_sample_old(alpha):
    n = len(alpha)
    y = np.zeros(n)
    ys = 0
    for i in range(n):
        y[i] = np.random.gamma(shape=alpha[i],scale=1)
        ys += y[i]
    y = y / ys
    return y

@njit(parallel=False,fastmath=True)
def log_pdf_dirichlet(x,alpha):
    l_beta_top = 0
    l_beta_bot = 0
    frac = 0
    for _a,_x in zip(alpha,x):
        l_beta_top += math.lgamma(_a)
        l_beta_bot += _a
        frac += (_a - 1)*math.log(_x)
    l_beta_bot = math.lgamma(l_beta_bot)
    return frac + l_beta_bot - l_beta_top

@njit(parallel=False)
def dirichlet_sample(alpha):
    n = len(alpha)
    y = np.zeros(n)
    ys = 0
    for i in range(n):
        y[i] = np.random.gamma(shape=alpha[i],scale=1)
        ys += y[i]
    y = y / ys
    return y

@njit(parallel=False)
def sample_theta(spot, theta, dt, indx, dist, edge_influence, K, n_iter = 10):
        
    alpha = dt[spot,:]
    
    theta_mat = np.zeros((n_iter+1, K))
        
    theta_mat[0,:] = theta[spot,:]
    old_log_proposal = log_pdf_dirichlet(theta_mat[0,:], alpha)
    #old_log_proposal = st.dirichlet(alpha).logpdf(theta_mat[0,:])
    old_log_target = log_potential(edge_influence, theta_mat[0,:], theta[indx[spot,:]])
    old_log_target = old_log_target * dist[spot,:] # Keep only those that are real neighbours. * 0 cancels false neighbours.
    old_log_target = old_log_target.sum()
    
    for it in range(1,n_iter+1):
        new_theta = dirichlet_sample(alpha)
        new_log_proposal = log_pdf_dirichlet(new_theta, alpha)
        #new_log_proposal = st.dirichlet(alpha).logpdf(new_theta)

        new_log_target = log_potential(edge_influence, new_theta, theta[indx[spot,:]])
        new_log_target = new_log_target * dist[spot,:]
        new_log_target = new_log_target.sum()
        
        log_u = new_log_target + old_log_proposal - old_log_target - new_log_proposal
 #       logu = astype(dtype = np.float128)
        u = np.exp(log_u) 
        a = min(u,1)
        
        if np.random.random() < a:
            theta_mat[it,:] = new_theta
            old_log_target = new_log_target
            old_log_proposal = new_log_proposal
        else:
            theta_mat[it,:] = theta_mat[it-1,:]
                    
#        theta[spot,:] = theta_mat[6:-1, :].mean(axis=0)
        #theta[spot,:] = theta_mat.mean(axis=0)
        theta[spot,:] = theta_mat[-1,:]
        

def sample_theta_old(spot, theta, dt, indx, dist, edge_influence, K, n_iter = 10):
        
    proposal_dist = st.dirichlet(dt[spot,:])
        
    theta_mat = np.zeros((n_iter+1, K))
        
    theta_mat[0,:] = theta[spot,:]
    old_log_proposal = proposal_dist.logpdf(theta_mat[0,:])
    old_log_target = log_potential(edge_influence, theta_mat[0,:], theta[indx[spot,:]])
    old_log_target *= dist[spot,:] # Keep only those that are real neighbours. * 0 cancels false neighbours.
    old_log_target = old_log_target.sum()
    
    for it in range(1,n_iter+1):
        new_theta = proposal_dist.rvs()[0]
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

def extract_data(phi,
                 theta,
                 gene_names,
                 tag,
                 ):
    
    phi_df = pd.DataFrame(phi,
                          columns = gene_names,
                          index = [f"topic_{k}" for k in range(phi.shape[0])],
                          )
    theta_df = pd.DataFrame(theta,
                          #columns = ,
                          #index = [f"topic_{k}" for k in range(theta.shape[1])],
                          )
    
    phi_path = tag + "-" + "phi.tsv"
    theta_path = tag + "-" + "theta.tsv"
    
    phi_df.to_csv(phi_path, sep = "\t")
    theta_df.to_csv(theta_path, sep = "\t")
    
def timestamp() -> str:
    """
    Helper function to generate a
    timestamp.

    Returns:
    -------
    String representing the timestamp
    """
    return re.sub(':|-|\.| |','',
                  str(datetime.datetime.now()))
    
def correlate_factors(phi_obs, phi_true, theta_true):
    cos_dist = cdist(phi_obs, phi_true, 'cosine')
    factor_order = cos_dist.argmin(axis=1)
    phi_true_ord = phi_true[factor_order, :]
    theta_true_ord = theta_true[:, factor_order]
    return phi_true_ord, theta_true_ord

def pearson(theta, theta_true_ord, K):
    pcor = np.zeros((K, 2))
    for i, t in enumerate(theta):
        pcor[i] = st.pearsonr(t, theta_true_ord[i])
    return pcor
 #   means = np.absolute(pcor).mean(axis=0)
 #   pearson_mean =  means[0]
 #   pval_mean = means[1]
 #   return pearson_mean, pval_mean

def validate(phi, theta, adata, K):
    # align factors for true phi and theta
    phi_true = adata.varm['phi'].T
    theta_true = adata.obsm['theta']
    phi_true_ord, theta_true_ord = correlate_factors(phi, phi_true, theta_true)
    
    # get commercial model prediction and align factors
    model = lda.LDA(n_topics=3, n_iter=1500, random_state=1)
    X = np.array(adata.X, dtype='int64')
    model.fit(X)
    phi_pipy, theta_pipy = correlate_factors(phi, model.topic_word_, model.doc_topic_)
    
    # pearson correlation
    MRTF_pc = pearson(theta.T, theta_true_ord.T, K)
    pipy_pc = pearson(theta_pipy.T, theta_true_ord.T, K)
#    MRTF_pear_mean, MRTF_pval_mean = pearson(theta.T, theta_true_ord.T, n_spots, K)
#    pipy_pear_mean, pipy_pval_mean = pearson(theta_pipy.T, theta_true_ord.T, n_spots, K)
    
    # print results
    print(MRTF_pc)
    print(pipy_pc)
 #   print('MRTF')
 #   print('Topic0, cor: {}, pval: {}'.format(MRTF_pc[]))
#    print('MRTF pearsons cor coef: {}'.format(MRTF_pear_mean))
#    print('PiPy pearsons cor coef: {}'.format(pipy_pear_mean))
#    print('MRTF p-value mean: {}'.format(MRTF_pval_mean))
#    print('PiPy p-value mean: {}'.format(pipy_pval_mean))

def plot_theta_synth(adata, dt, theta, tag):
    theta = get_theta(dt)
    side_length = adata.uns["info"]["side_length"]
    plt.imshow(theta.reshape(side_length,side_length,3))
    plt.title(tag)
    plt.show()

def plot_theta(adata, dt, theta, tag, s=5):

    theta = get_theta(dt)
    crd = adata.obsm["spatial"]
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(crd[:,1],crd[:,0],c = theta,s=s,edgecolor = "none")
    ax.set_aspect("equal")
    for sp in ax.spines.keys():
        ax.spines[sp].set_visible(False)       
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(tag)
    plt.show()


