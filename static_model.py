from network_propagation import SquareLattice
import cplex
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def main():

    # start cplex
    c = cplex.Cplex()
    c.set_problem_type(c.problem_type.LP)
    c.objective.set_sense(c.objectvie.sense.minimize)

    # add variables and objective function
    
