#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:03:37 2021

@author: juliafoyer
"""

import argparse
import random
import numpy as np
import scipy.stats

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
    # The words_alpha.txt file contains 370103 words in total. Keep only 10,000 random words.
    words_list = []
    with open("words_alpha.txt", 'r') as file:
        for line in file:
            line = line.strip("\n")
            words_list.append(line)
    words_list = random.sample(words_list, args.words)
    
    # Prepare a file for the documents and a file for metadata.    
    D = open('corpus.txt', 'w')
    metafile = open('metainformation.txt', 'w')
    
    # Chose number of documents (M), number of topics (K), and number of words for each document (N).
    M = args.documents
    K = args.topics
    N = np.random.randint(100, 200, size = M)
    
    # Create the alpha parameter vector. Each column is a topic. Dimensions 1 x K.
    alpha = args.alpha * np.ones(K)
    
    # Create the beta parameter vector. Each column is a word. Dimensions 1 x len(words_list).
    beta = args.beta * np.ones(len(words_list))
    
    # For each document, chose theta. Dimensions M x K.
    theta = np.random.dirichlet(alpha, size = K)
       
    # For each document, chose phi. Dimensions M x W.
    phi = np.random.dirichlet(beta, size = M)
      
    # Inte klar. For each word position in each document, chose a topic (z).
    topic_dict = {}
    
    # Inte klar. For each word position in each document, chose a word (w).
    i = 0
    while i < M:
        j = 0
        while j < N[i]:
            w = scipy.stats.multinomial()
            D.write(str(w))
            D.write(" ")
            j += 1
        D.write("\n")
        i += 1
        
    
    
        
        
 #       N = random.randint(500, 800)
 #       for position in range(0, N):
 #           z = scipy.stats.multinomial(1, theta)
 #           metafile.write('{}\n'.format(z))
 #           w = scipy.stats.multinomial(1, phi)
#            D.write('{} ').format(w)

if __name__ == '__main__':
    main()