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
beta = 0.1
Gibbs_iterations = 50

path = "/Users/juliafoyer/Documents/Skolarbete/Masters_thesis/Scripts/spatial-data-synth/20210402100654897026-synth-data.h5ad"
adata = an.read_h5ad(path)
adata = prepare_data(adata,select_hvg=False)

umi_factors = assign_random(adata, K)

n_spots,n_genes = adata.shape
ids,dt,wt = get_ids_dt_wt(adata,umi_factors, K)#, alpha = alpha, beta = beta)
nz = get_nz(dt)

#gibbsSampling(umi_factors, ids, dt, wt, nz, Gibbs_iterations)

#Gibbs sampling
for it in range(Gibbs_iterations):
    for d, doc in enumerate(ids):
        for index, w in enumerate(doc):
            draw_new_factor(umi_factors, dt, wt, nz, d, index, w)
    
plot_theta(adata, dt)