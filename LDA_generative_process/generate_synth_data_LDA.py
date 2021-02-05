#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:03:37 2021

@author: juliafoyer
"""

import argparse
import random
import numpy as np

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
    
    args = parser.parse_args()
      
    # Read the file containing words, separate into words in a list.
    words_list = []
    with open("words_alpha.txt", 'r') as file:
        for line in file:
            line = line.strip("\n")
            words_list.append(line)
    words_list = random.sample(words_list, args.words)
    
    # Prepare a file for the documents. 
    D = open('corpus.txt', 'w')
    
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
      
    # For each word position (j) in each document (i), chose a topic (z) and a word (w).
    topic_dict = {}
    i = 0
    while i < M:
        topic_dict[i] = []
        j = 0
        while j < N[i]:
            z_array = np.random.multinomial(1, theta[i])
            z_lst = z_array.tolist()
            z = z_lst.index(1)
            topic_dict[i].append(z)
            w_array = np.random.multinomial(1, phi[z])
            w_lst = w_array.tolist()
            w_ind = w_lst.index(1)
            w = words_list[w_ind]
            D.write(w)
            D.write(" ")
            j += 1
        D.write("\n")
        i += 1
    
if __name__ == '__main__':
    main()