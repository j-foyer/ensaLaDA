#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:48:20 2021

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

def get_gene_identity(adata):
    """ add short description here
    
    Parameters:
    ----------
    """
    n_spots,n_genes = adata.shape
    gene_id_list = []
    X = adata.X
    for spot in range(n_spots):
        gene_id_list.append([])
        for gene in range(n_genes):
            gene_id_list[spot] += [gene] *int(X[spot,gene])
    return gene_id_list

def get_ids_dt_wt_test(gene_ids: List[List[int]],
                  umi_factors: List[np.ndarray],
                  n_genes: int,
                  K: int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ add short description here
    
    Parameters:
    ----------
    """
    
    n_spots = len(gene_ids)
    dt = np.zeros((n_spots, K))
    wt = np.zeros((K, n_genes))
    
    for spot in range(n_spots):
        for gene,factor in zip(gene_ids[spot],umi_factors[spot]):
            dt[spot,factor] += 1
            wt[factor,gene] += 1
    
    return dt,wt

def get_theta_test(dt):
    """ add short description here
    
    Parameters:
    ----------
    """
    n_spots, K = dt.shape
    theta = np.zeros((n_spots, K))
    for spot in range(n_spots):
        for factor in range(K):
            theta[spot, factor] = dt[spot, factor] / np.sum(dt, axis=1)[spot]
    return theta

def remove_false_neighbours_test(dist,indx):
    """ add short description here
    
    Parameters:
    ----------
    """
    nbr_filter = lambda xs,ds : [x for x,d in zip(xs,ds) if not np.isinf(d)]
    new_idx = [nbr_filter(i,d) for i,d in zip(indx,dist)]
    new_dist = [nbr_filter(d,d) for d in dist]
    return new_dist,new_idx

def get_E_test(indx_sel: List[int])->int:
    """ add short description here
    
    Parameters:
    ----------
    """
    # make list to hold neighbor index pairs
    indx_tuples = []
    # iterate over neibhorhoods
    for spot,nbrhd in enumerate(indx_sel):
        # iterate over neihbors in neighborhood
        for nbr in nbrhd[1::]:
            # make a pair (spot,neibhor)
            pair = [spot,nbr]
            # sort pair to store edges the same way
            # (a,b) and (b,a) will now be (a,b) and (a,b)
            pair.sort()
            # store pair, convert to tuple
            indx_tuples.append(tuple(pair))
    # apply set to pair list to remove duplicates
    # will only hold unique edges now
    indx_tuples = set(indx_tuples)
    # return length
    return len(indx_tuples)

