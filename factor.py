from network_propagation import SquareLattice
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from scipy.stats import describe
from itertools import product

def psi_node(a, beta=1):
    global sigma, h
    return np.exp(-beta*sigma[a])*(1+np.exp(h[a])*(sigma[a]==1))/(1+np.exp(h[a]))

def psi_node_deterministic(a, beta=1):
    global sigma, sigma_in
    return np.exp(-beta*sigma[a])*(sigma[a]>=sigma_in[a]))

def psi_edge(a,b):
    global sigma, g
    return (1+np.exp(g[a,b])*(sigma[a]==sigma[b]))/(1+np.exp(g[a,b]))

# need sigma, h, and g
n = 10
sq = SquareLattice(n)

# binary vectors of initially infected and finally recovered
sigma_in, sigma = sq.dynamic_process(shuffle=1, view=0)
# # parameter determining probability of initially infected
# h = sigma_in
# extract active edge weights
g, thetas = sq.get_active_edge_weights()


# a bucket for node i is a set of factors operating on node i
# i.e. the bucket consists of factors for node a and its neighbors
# Generate a new factor by (1)
