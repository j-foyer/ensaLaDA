import numpy as np
import pandas as pd
import argparse as arp
import anndata as ad




# Read data  With argparse


adata = ad.read_h5ad(args.count_data)

findKNN(adata,k = args.k)


