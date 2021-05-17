#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:44:18 2021

@author: juliafoyer
"""

from time import perf_counter
start = perf_counter()

import anndata as an
import argparse
import os
import datetime
from tqdm import tqdm

from ST_LDA_functions import *
from test_ST_LDA_functions import *

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg("-n", "--neighbours",
        type = int,
        default = 6,
        help = "Specify number of neighbours."\
            " If not specified number of neighbours is 6.")

add_arg("-a", "--alpha",
        type = float,
        default = 0.1,
        help = "Specify alpha as scalar."\
            " If not specified alpha is 5.0.")

add_arg("-b", "--beta",
        type = float,
        default = 0.1,
        help = "Specify beta as scalar."\
            " If not specified alpha is 0.1.")
    
add_arg("-f", "--factors",
        type = int,
        default = 3,
        help = "Specify number of factors."\
            " If not specified there will be 3 factors.")

add_arg("-it", "--iterations",
        type = int,
        default = 150,
        help = "Specify number of Gibbs iterations."\
            " Default is 150 iterations.")

add_arg("-w", "--weight",
        type = int,
        default = 50,
        help = "Specify how much weight the neighbours"\
            " should have on theta. Default is 50.")

add_arg("-ss", "--subsample",
        default=False,
        help = "Write True if you want to subsample"\
            "the data for faster performance.")

add_arg("-df", "--dataformat", 
        default="visium", # change to visium later
        help = "Specify if data format is visium or h5ad."\
            "Default is visium. Otherwise, type h5ad.")

add_arg("-p", "--path",
        help = "Give the path to the data"\
            "you want to analyze.")
    
add_arg("-d", "--directory", type = str,
        help = "Give a directory where you want"\
            " the files to be saved."\
                " Default is current working directory.")

add_arg("-t","--tag",
        type = str,
        default = None,
        help = "tag for output files.")
    
args = parser.parse_args()

# Set the output directory.
if args.directory is not None:
    path = args.directory
    if not os.path.isdir(path):
        os.makedirs(path)

# Prepare tag for output files
if args.tag is not None:
    tag = args.tag
else:
    tag = timestamp()
   
# Prepare paths for files
genes_path = tag + "-" + "top_genes.txt"
genes_phi_path = tag + "-" + "top_genes_phi.npy"

if args.directory is not None:
    genes_path = os.path.join(directory, genes_path)
    genes_phi_path = os.path.join(directory, genes_phi_path)

K = args.factors
alpha = args.alpha
beta = args.beta
Gibbs_iterations = args.iterations
edge_influence = args.weight
n_neighbours = args.neighbours
path = args.path

if args.dataformat == "synth":
    adata = an.read_h5ad(path)
    adata = prepare_data(adata, todense = False, select_hvg = False)

elif args.dataformat == "visium":
    adata = sc.read_visium(path)
    adata = prepare_data(adata)

elif args.dataformat == "tsv":
    counts_path = "/Users/juliafoyer/Documents/Skolarbete/Masters_thesis/her2st_G2/G2.tsv"
    metrics_path = "/Users/juliafoyer/Documents/Skolarbete/Masters_thesis/her2st_G2/G2_selection.tsv"
    adata = create_object(counts_path, metrics_path)
    adata = prepare_data(adata, todense = False)

else:
    print("The data format given is not supported")
    
if args.subsample == "True":
    adata = subsample(adata)

# Assign random factors to start, and get count matrices.
n_spots, n_genes = adata.X.shape
umi_factors = assign_less_random(adata, K)
ids,dt,wt = get_ids_dt_wt(adata,umi_factors, K)
nz = get_nz(wt, beta, n_genes)
theta = get_theta(dt)

# Prepare graph
dist_sel, indx_sel = build_nbrhd(adata, n_neighbours)

# Gibbs sampling
for it in tqdm(range(Gibbs_iterations)):
    for d, doc in enumerate(ids): # loop through each spot
        for index, w in enumerate(doc): # loop through each umi
            umi_factors[d] = draw_new_factor(umi_factors[d], dt, wt, nz, theta, d, index, w, beta, n_genes)
            sample_theta(d, theta, dt, indx_sel, dist_sel, edge_influence, K)
 
    end = perf_counter()
    #print("\riteration", it,end="")#, round((end-start)/60, 1), "minutes")    

end = perf_counter()
print("Finished in", round((end-start)/60, 1), "minutes")

phi = get_phi(wt)
theta = get_theta(dt)


gene_names = adata.var_names
extract_data(phi, theta, gene_names, tag)     



