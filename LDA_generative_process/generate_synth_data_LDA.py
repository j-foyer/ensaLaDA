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

# Needs to be fixed:
    # Make sure alpha and beta are in the right format. If they are wrong, theta and phi is also wrong.
    # I'm not sure what I expected for z (topics).
    # I'm not getting words (w) yet when I sample from the multinomial distribution.
    # The code should be changed from hard coded so that some variables and file names can be given as CL arguments.

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    args = parser.parse_args()
    
#    add_arg('words',
#            type = str)
      
    # Read the file containing words, separate into words in a list.
    # The words_alpha.txt file contains 370103 words in total.
    words_list = []
    with open("words_alpha.txt", 'r') as file:
        for line in file:
            line = line.strip("\n")
            words_list.append(line)
    
    # Prepare a file for the documents and a file for metadata.    
    D = open('corpus.txt', 'w')
    metafile = open('metainformation.txt', 'w')
    
    # Chose number of documents (M) and number of topics (K).
    M = random.randint(100, 200)
    K = random.randint(20, 40)
    
    # Create a document and write it to the corpus file.
    # Write metainfo to the metafile.
    alpha = np.array([0.2, 0.5, 0.8])
    beta = np.array([0.2, 0.5, 0.8])
           
    # For each topic, chose phi.
    phi = np.random.dirichlet(beta)
    
    # For each document, chose theta.
    for document in range(0, 1):
        theta = np.random.dirichlet(alpha)
        
        # Store theta and phi in metadata file.
        metafile.write('{}\t{}\t'.format(theta, phi))
        
        # For each word position, chose a topic (z) and a word (w).
        j = random.randint(500, 800)
        for position in range(0, j):
            z = scipy.stats.multinomial(1, theta)
            metafile.write('{}\n'.format(z))
            w = scipy.stats.multinomial(1, phi)
            D.write('{} ').format(w)

if __name__ == '__main__':
    main()