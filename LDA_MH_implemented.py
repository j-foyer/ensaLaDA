#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:44:18 2021

@author: juliafoyer
"""

import anndata as an

from ST_LDA_functions import *
from test_ST_LDA_functions import *

K = 3
alpha = 5
alpha_orig = np.ones(K)
beta = 0.1
Gibbs_iterations = 2

path = "/Users/juliafoyer/Documents/Skolarbete/Masters_thesis/Scripts/spatial-data-synth/20210402100654897026-synth-data.h5ad"
adata = an.read_h5ad(path)
adata = prepare_data(adata,select_hvg=False)

umi_factors = assign_random(adata, K)

n_spots,n_genes = adata.shape
ids,dt,wt = get_ids_dt_wt(adata,umi_factors, K)#, alpha = alpha, beta = beta)
nz = get_nz(dt)
theta = get_theta(dt)

# Prepare graph
dist, indx = findKNN(adata, 4, max_distance = 1.1) # Can I define it at the top?
dist_sel, indx_sel = remove_false_neighbours(dist, indx)
E = get_E(indx_sel)

# Get first lambda and edge_influence estimations
lambda0 = float(np.random.gamma(0.01, 0.01, 1)) # The 1 seems to be unnecessary here.
edge_influence = first_edge_influence(lambda0, theta, indx_sel, K)
lambda_parameter = sample_lambda(E, edge_influence)

# Gibbs sampling
for it in range(Gibbs_iterations):
    for d, doc in enumerate(ids): # loop through each spot
        for index, w in enumerate(doc): # loop through each umi
            draw_new_factor(umi_factors, dt, wt, nz, d, index, w)
            theta = get_theta(dt)
  #          sample_edge_influence(edge_influence, lambda_parameter, theta, indx_sel, K) # set 1 or 0.1
  #          lambda_parameter = sample_lambda(E, edge_influence)
  #          metropolis_hastings(d, theta, dt, indx_sel, alpha_orig, edge_influence)
            


plot_theta(adata, dt)