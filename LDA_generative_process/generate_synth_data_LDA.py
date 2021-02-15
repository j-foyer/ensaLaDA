#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:03:37 2021

@author: juliafoyer
"""

import argparse
import random
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg("-a", "--alpha", type = float, default = 0.1,
            help = "Specify alpha as scalar. If not specified alpha is 0.1.")
    
    add_arg("-b", "--beta", type = float, default = 0.1,
            help = "Specify beta as scalar. If not specified beta is 0.1.")
    
    add_arg("-w", "--words", type = int, default = 1000,
            help = "The total number of unique words in the corpus. Default is 1000.")
    
    add_arg("-M", "--documents", type = int, default = 50,
            help = "The total number of documents in the corpus. Default is 50.")
    
    add_arg("-K", "--topics", type = int, default = 20,
            help = "The total number of topics. Default is 20.")
    
    add_arg("-d", "--directory", type = str,
             help = "Give a directory where you want the files to be saved. Default is current working directory.")
    
    args = parser.parse_args()
      
    # Read the file containing words, separate into words in a list.
    words_list = []
    with open("words_alpha.txt", 'r') as file:
        for line in file:
            line = line.strip("\n")
            words_list.append(line)
    words_list = random.sample(words_list, args.words)
    
    # Set the output directory.
    if args.directory is not None:    
        path = args.directory
        if not os.path.isdir(path):
            os.makedirs(path)    
    
    # Prepare files for the documents, theta and phi.
    corpuspath = "corpus.txt"
    thetapath = "theta.npy"
    phipath = "phi"
    if args.directory is not None:    
        corpuspath = os.path.join(path, corpuspath)
        thetapath = os.path.join(path, thetapath)
        phipath = os.path.join(path, phipath)
    D = open(corpuspath, 'w')
 
    # Chose number of documents (M), number of topics (K), and number of words for each document (N).
    M = args.documents
    K = args.topics
    N = np.random.randint(100, 200, size = M)
    W = len(words_list)
    
    # Create the alpha parameter vector. Dimensions 1 x K.
    alpha = args.alpha * np.ones(K)
    
    # Create the beta parameter vector. Dimensions 1 x W.
    beta = args.beta * np.ones(W)
    
    # For each document, chose theta. Dimensions M x K.
    theta = np.random.dirichlet(alpha, size = M)
    
    # For each topic, chose phi. Dimensions K x W.
    phi = np.random.dirichlet(beta, size = K)
    phi_dict = {}
      
    # For each word position (j) in each document (i), chose a topic (z) and a word (w).
    for document in range(M):
        phi_dict[document] = np.empty((0,W), float)
        for position in range(N[document]):
            z_array = np.random.multinomial(1, theta[document])
            z = np.argmax(z_array)
            phi_dict[document] = np.vstack([phi_dict[document], phi[z]])
            w_array = np.random.multinomial(1, phi[z])
            w_ind = np.argmax(w_array)
            w = words_list[w_ind]
            D.write(w)
            D.write(" ")
        D.write("\n")
      
    # Save all theta and phi values.    
    np.save(thetapath, theta)
#    np.savez(phipath, phi_dict)
    
    for key in phi_dict:
        np.save("{}_doc{}".format(phipath, key), phi_dict[key])
    
if __name__ == '__main__':
    main()