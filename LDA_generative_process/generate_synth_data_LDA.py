#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:03:37 2021

@author: juliafoyer
"""
import os
import argparse
import random
import numpy as np

import re
import datetime


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

    add_arg("-w", "--words",
            type = int,
            default = 1000,
            help = "The total number of unique"\
            " words in the corpus. Default is 1000.")

    add_arg("-M", "--documents", type = int, default = 50,
            help = "The total number of documents"\
            " in the corpus. Default is 50.")

    add_arg("-K", "--topics", type = int, default = 20,
            help = "The total number of topics."\
            " Default is 20.")

    add_arg("-d", "--directory", type = str,
             help = "Give a directory where you want"\
            " the files to be saved."\
            " Default is current working directory.")

    add_arg("-t","--tag",
            type = str,
            default = None,
            help = "tag to prepend output files with",
            )

    args = parser.parse_args()

    if args.tag is not None:
        tag = args.tag
    else:
        tag = timestamp()

    # Read the file containing words, separate into words in a list.
    words_list = []
    with open("words_alpha.txt", 'r') as file:
        for line in file:
            line = line.strip("\n")
            words_list.append(line)

    # TODO: there's an issue here, the random.sample function
    # allows for resampling, meaning we can get the same word twice
    # I would recommend yoy to use the numpy.random.choice function with replace = False
    words_list = random.sample(words_list, args.words)

    # Set the output directory.
    if args.directory is not None:
        path = args.directory
        if not os.path.isdir(path):
            os.makedirs(path)

    # Number of documents
    M = args.documents
    # Number of topics
    K = args.topics
    # number of words in each document
    N = np.random.randint(100, 200, size=M)
    # number of words in curpus
    W = len(words_list)

    # Create the alpha parameter vector. Dimensions 1 x K.
    alpha = args.alpha * np.ones(K)

    # Create the beta parameter vector. Dimensions 1 x W.
    beta = args.beta * np.ones(W)

    # For each document, chose theta. Dimensions M x K.
    theta = np.random.dirichlet(alpha, size=M)

    # For each topic, chose phi. Dimensions K x W.
    # TODO: This is the phi information that you want to save
    phi = np.random.dirichlet(beta, size=K)


    # corpus basename
    corpuspath = tag + "-" + "corpus.txt"
    # theta basename
    thetapath = tag + "-" + "theta.npy"
    #phi basename
    phipath = tag + "-" + "phi.npy"

    # set full path if output dir is provided
    if args.directory is not None:
        corpuspath = os.path.join(path,corpuspath)
        thetapath = os.path.join(path, thetapath)
        phipath = os.path.join(path, phipath)

    # save parameters
    np.save(phipath,phi)
    np.save(thetapath,theta)


    # TODO: add a comment here what phi_dict is; this can be removed
    # phi_dict = {}

    # open stream
    D = open(corpuspath, 'w')

    for document in range(M):
        # phi_dict[document] = np.empty((0,W), float)
        for position in range(N[document]):
            z_array = np.random.multinomial(1, theta[document])
            z = np.argmax(z_array)
            # phi_dict[document] = np.vstack([phi_dict[document], phi[z]])
            w_array = np.random.multinomial(1, phi[z])
            w_ind = np.argmax(w_array)
            w = words_list[w_ind]
            D.write(w)
            D.write(" ")
        D.write("\n")

    # make sure to close stream
    D.close()


    # Save all theta and phi values.    
    # np.save(thetapath, theta)
    # np.savez(phipath, phi_dict)

    # for key in phi_dict:
        # np.save("{}_doc{}".format(phipath, key), phi_dict[key])

if __name__ == '__main__':
    main()
