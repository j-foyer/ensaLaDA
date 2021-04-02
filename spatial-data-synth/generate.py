#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:03:37 2021

@author: juliafoyer
"""
import os
import os.path as osp
import argparse
import random
import numpy as np
import anndata as ad
from scipy.spatial import KDTree
import pandas as pd

import re
import datetime

def remove_false_neighbours(dist,indx):
    nbr_filter = lambda xs,ds : [x for x,d in zip(xs[1::],ds[1::]) if not np.isinf(d)]
    new_idx = [nbr_filter(i,d) for i,d in zip(indx,dist)]
    new_dist = [nbr_filter(d,d) for d in dist]
    return new_dist,new_idx


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

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("-a", "--alpha",
            type = float,
            default = 0.1,
            help = "Specify alpha as scalar."\
            " If not specified alpha is 0.1.")

    add_arg("-b", "--beta",
            type = float,
            default = 0.1,
            help = "Specify beta as scalar."\
            " If not specified beta is 0.1.")

    add_arg("-g", "--genes",
            type = int,
            default = 50,
            help = "The total number of unique"\
            " words in the corpus. Default is 1000.")

    add_arg("-sl","--array_side_length",
            type = int,
            default = 10,
            help = "length of array sides. Number of"\
            " spots will be side_length**2",
            )


    add_arg("-K", "--factors", type = int, default = 3,
            help = "The total number of factors."\
            " Default is 20.")

    add_arg("-o", "--out_dir", type = str,
             help = "Give a directory where you want"\
            " the files to be saved."\
            " Default is current working directory.")

    add_arg("-t","--tag",
            type = str,
            default = None,
            help = "tag to prepend output files with",
            )

    add_arg("-si","--smoothing_iterations",
            type = int,
            default = 10,
            )

    add_arg("-wul","--words_upper_limit",
            type = int,
            default = 200,
            )


    args = parser.parse_args()

    if args.tag is not None:
        tag = args.tag
    else:
        tag = timestamp()


    xs = np.arange(args.array_side_length)
    ys = np.arange(args.array_side_length)
    xs,ys = np.meshgrid(xs,ys)
    xs = xs.flatten()
    ys = ys.flatten()
    crd = np.hstack((xs[:,np.newaxis],ys[:,np.newaxis]))


    kd = KDTree(crd)
    dist,indx = kd.query(crd,
                         k = 5,
                         distance_upper_bound=np.sqrt(2),
                         eps = 0.2)

    dist,indx = remove_false_neighbours(dist,indx)
    dist,indx = remove_false_neighbours(dist,indx)

    # Set the output directory.
    if args.out_dir is not None:
        path = args.out_dir
        if not os.path.isdir(path):
            os.makedirs(path)

    # Number of documents
    S = len(crd)
    # Number of topics
    K = args.factors
    # number of words in each document
    N = np.random.randint(10, args.words_upper_limit, size=S)
    # number of available genes
    G = args.genes
    # Create the alpha parameter vector. Dimensions 1 x K.
    alpha = args.alpha * np.ones(K)

    # Create the beta parameter vector. Dimensions 1 x W.
    beta = args.beta * np.ones(G)

    # For each document, chose theta. Dimensions M x K.
    theta = np.random.dirichlet(alpha, size=S)

    for it in range(args.smoothing_iterations):
        gamma = np.exp(-2 + it/args.smoothing_iterations)
        for s in range(S):
            nbrs = indx[s]
            av_nbr_theta = theta[nbrs,:].mean(axis=0)
            theta[s,:] = gamma * av_nbr_theta + (1-gamma)*theta[s,:]

    assert round(theta.sum()) == S, "theta values do not sum to one"


    phi = np.random.dirichlet(beta, size=K)
    counts = np.zeros((S,G))
    for s in range(S):
        # phi_dict[document] = np.empty((0,W), float)
        for n in range(N[s]):
            z = np.argmax(np.random.multinomial(1,theta[s,:]))
            w = np.argmax(np.random.multinomial(1,phi[:,z]))
            counts[s,w] += 1


    var_idx = ["Gene_{}".format(x) for x in range(G)]
    obs_idx = ["Spot_{}".format(x) for x in range(S)]
    var = pd.DataFrame(var_idx,index = var_idx, columns = ["gene"])
    obs = pd.DataFrame(obs_idx,index = obs_idx, columns = ["spot"])

    adata = ad.AnnData(counts,
                       obs = obs,
                       var = var,
                       )

    adata.obsm["theta"] = theta
    adata.obsm["spatial"] = crd
    adata.varm["phi"] = phi.T
    adata.uns["info"] = dict(side_length = args.array_side_length,
                             smoothing_iterations = args.smoothing_iterations,
                             alpha = alpha,
                             beta = beta,
                             )

    adata.write_h5ad(osp.join(args.out_dir,tag + "-synth-data.h5ad"))

if __name__ == '__main__':
    main()
