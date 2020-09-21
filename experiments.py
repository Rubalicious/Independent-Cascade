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

def dynamic_process(sq, shuffle=0):
    sq.infect([90])
    # sq.view_current_state()
    A = sq.sample_active_edges()
    # sq.view_active_edges(A)
    while not sq.is_totally_infected():
        sq.update(A, shuffle)
    v = sq.build_state_vector()/2
    # sq.view_current_state()
    sq.reset()
    return A, v

def static_process_exp(sq):
    sq.infect([90])
    # sq.view_current_state()
    A = sq.sample_active_edges()
    sq.static_process(A)
    # sq.view_current_state()
    v = sq.build_state_vector()
    sq.reset()
    return A,v

def static_vs_dynamic():
    n = 13
    sq = SquareLattice(n)

    # dynamic process
    A,v = dynamic_process(sq,1)
    # A,v = static_process_exp(sq)
    # N = 1
    # for i in range(N):
    #     A,w = dynamic_process(sq, 1)
    #     # A,w = static_process_exp(sq)
    #     sys.stdout.write('\r')
    #     sys.stdout.write('{}% complete'.format(100*float(i)/N))
    #     sys.stdout.flush()
    #     v+=w
    # print(v/N)
    # v = np.array(v)
    # v = np.reshape(v,(n,n+1))
    # plt.imshow(v/N, cmap='hot')
    # plt.colorbar()
    # # plt.plot(range(sq.n*sq.m), v/N)
    # plt.show()

def influence_of_infection_order():
    '''
        Compute the average of final configurations with and without
        shuffled order of infections
        Computing the probability of a node becoming infected
    '''
    n = 17
    sq = SquareLattice(n)

    N = 200 # number of trials

    shuffle = 1
    def stats(shuffle=1):
        if shuffle:
            print('computing with shuffle ...')
        else:
            print('computing without shuffle ...')

        data = np.zeros((n*(n+1),N))
        for i in range(N):
            data[:,i] = np.reshape(sq.dynamic_process(shuffle), (sq.N))
            sys.stdout.write('\r')
            sys.stdout.write('{}% complete'.format(100*float(i+1)/N))
            sys.stdout.flush()
        print('')
        return data

    s_data = stats(shuffle=1)
    s_nobs, s_minmax, s_mean, s_var, s_skew, s_kurt = describe(s_data, axis=1)
    # print(s_nobs, s_minmax, s_mean, s_var, s_skew, s_kurt)
    o_data = stats(shuffle=0)
    o_nobs, o_minmax, o_mean, o_var, o_skew, o_kurt = describe(o_data, axis=1)
    # print(o_nobs, o_minmax, o_mean, o_var, o_skew, o_kurt)

    # KL Divergence
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    kl_data = np.zeros((n*(n+1)))
    for i in range(n*(n+1)):
        kl_data[i] = kl_divergence(s_data[i,:], o_data[i,:])
    kl_data = np.reshape(kl_data, (n,n+1))
    plt.imshow(kl_data, cmap='hot')
    plt.title('KL Divergence for each node')
    plt.colorbar()
    plt.show()

    # Error metrics
    abs_err_avg = np.abs(s_mean-o_mean)
    rel_err_avg = np.abs(s_mean-o_mean)/s_mean
    mod_err_avg = np.abs(s_mean-o_mean)**2/(s_mean*o_mean)

    abs_err_std = np.abs(s_var-o_var)
    rel_err_std = np.abs(s_var-o_var)/np.abs(s_var)
    mod_err_std = np.abs(s_var-o_var)**2/np.abs(s_var*o_var)

    abs_err_ske = np.abs(s_skew-o_skew)
    rel_err_ske = np.abs(s_skew-o_skew)/np.abs(s_skew)
    mod_err_ske = np.abs(s_skew-o_skew)**2/np.abs(s_skew*o_skew)

    abs_err_kur = np.abs(s_kurt-o_kurt)
    rel_err_kur = np.abs(s_kurt-o_kurt)/np.abs(s_kurt)
    mod_err_kur = np.abs(s_kurt-o_kurt)**2/np.abs(s_kurt*o_kurt)


    def plot_results(avg, var, skew, kurt, title):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        img = np.reshape(avg,(n,n+1))
        im1 = ax1.imshow(img, cmap='hot')
        ax1.set_title('mean')
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="10%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


        img = np.reshape(var,(n,n+1))
        im2 = ax2.imshow(img, cmap='hot')
        ax2.set_title('variance')
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="10%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

        img = np.reshape(skew,(n,n+1))
        im3 = ax3.imshow(img, cmap='hot')
        ax3.set_title('skewness')
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="10%", pad=0.05)
        cbar3 = plt.colorbar(im3, cax=cax3, format="%.2f")

        img = np.reshape(kurt,(n,n+1))
        im4 = ax4.imshow(img, cmap='hot')
        ax4.set_title('kurtosis')
        ax4.xaxis.set_visible(False)
        ax4.yaxis.set_visible(False)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="10%", pad=0.05)
        cbar4 = plt.colorbar(im4, cax=cax4, format="%.2f")

        fig.suptitle(title)
        plt.savefig(title.replace(' ', '-'))
        plt.show()

    # shuffle plots
    plot_results(s_mean, s_var, s_skew, s_kurt, 'active edges with shuffle')

    # ordered plots
    plot_results(s_mean, s_var, s_skew, s_kurt, 'active edges without shuffle')

    # absolute error plots
    plot_results(abs_err_avg, abs_err_std, abs_err_ske, abs_err_kur, 'absolute error plots')

    # relative error plots
    plot_results(rel_err_avg, rel_err_std, rel_err_ske, rel_err_kur, 'relative error plots')

    # modified error plots
    plot_results(mod_err_avg, mod_err_std, mod_err_ske, mod_err_kur, 'modified error plots')



def check_substeps():
    '''
    the following experiment ensures that the final configuration
    is independent of the sub-steps taken to get there.

    Given the same initial infected set and active edges,
    simulate propagation of infection for random substeps.
    '''
    n = 3
    sq = SquareLattice(n)

    shuffle = 1 # shuffles the order in which infected nodes infect susceptible
    sq.infect([6])
    sq.view_current_state()
    A = sq.sample_active_edges()
    sq.view_active_edges(A)
    while not sq.is_totally_infected():
        sq.update(A, shuffle)

    # fig.savefig('final_config1.png')
    sq.reset()

    #repeat
    sq.infect([6])
    sq.view_current_state()
    sq.view_active_edges(A)
    while not sq.is_totally_infected():
        sq.update(A, shuffle)
    # plt.savefig('final_config2.png')

def main():
    '''
        check that we can hone in probabilities theta of the graph
        next, adapt for infectiousness for multiple time steps.
    '''
    n = 13
    sq = SquareLattice(n)
    A = sq.sample_active_edges()
    g, thetas = sq.get_active_edge_weights()

    # generate a random number multiple times and count the number of times it
    # beats the weight. Averaging until I converge on the value
    N = sq.N
    R = 5000
    output = np.zeros((N,N))
    for i in range(R):
        test = np.random.uniform(0,1, (N,N))
        # t_symm = (test+test.T)/2
        output += test < thetas
    avg_out = output/R
    print(avg_out)
    print('')
    print(thetas)
    print(np.max(abs(thetas-avg_out)))



def plot_demo():
    '''
        Plotting examples
    '''
    n=5
    sq = SquareLattice(n)
    sq.plot_demo(3)

def example():
    '''
        Runs an example of a dynamic process
    '''
    n=31
    sq = SquareLattice(n)
    sq.dynamic_process(shuffle=1, view=1)

def MBR():
    '''
        An implementation of Mini Bucket Renormalization
        to compute the Partition function
    '''
    # first need to encode the factor function f_a(sigma_a)
    n = 10
    sq = SquareLattice(n)
    beta = 1
    # binary vectors of initially infected and finally recovered
    sigma_in, sigma = sq.dynamic_process(shuffle=1, view=0)
    # parameter determining probability of initially infected
    h = sigma_in
    # extract active edge weights
    g, thetas = sq.get_active_edge_weights()
    # binary variable of active edge weights
    var_sigma = 1*(thetas > 0)

    def factor(a):
        global sq, sigma_in, var_sigma
        v = sq.build_state_vector()

        term = np.exp(-beta*1)
        

def compute_partition_function():
    '''
        A brute force computation of the partition function.
        To do this, need to extract the vectors sigma, sigma^(in), and var_sigma
    '''
    # Note: don't necessarily need the dynamic process
    n = 2
    sq = SquareLattice(n)
    beta = 1
    # binary vectors of initially infected and finally recovered
    sigma_in, sigma = sq.dynamic_process(shuffle=1, view=0)
    # parameter determining probability of initially infected
    h = sigma_in
    # extract active edge weights
    g, thetas = sq.get_active_edge_weights()
    # binary variable of active edge weights
    var_sigma = 1*(thetas > 0)

    # generate all possible initial infection scenarios
    all_initial_infections = list(product([0, 1], repeat=sq.N))
    print(all_initial_infections)
    # generate all possible active edge set scenarios

    def get_all_active_edge_sets(N = 10):
        all_active_edge_sets = []
        for k in range(N):
            for i in product([0,1], repeat = k):
                v = np.array(i)

        for i in product([0, 1], repeat = N*N):
            mat = np.reshape(np.array(i), (N, N))

            if (mat == mat.T).all():
                print(mat)
                all_active_edge_sets.append(mat)
        return all_active_edge_sets

    all_active_edge_sets = get_all_active_edge_sets()
    print(all_active_edge_sets)
    quit()

    # note: see if I can separate the sum in the partition function,
    # so that I can sum over initially infected nodes, and its neighbors
    # as opposed to suming over all initially infected nodes and all active edges.
    for a in range(sq.N):
        for b in sq.G.neighbors(a):
            # find if node is initially infected
            # if sq.G.nodes[a]['state'] == 1

            term = np.exp(sigma_in[a]*h[a]+np.sum(var_sigma[a,b]*g[a,b]))
            if sigma[a] >= sigma_in[a]: term = 0

    print(sigma >= sigma_in)
    print(np.matmul(var_sigma, sigma))

compute_partition_function()
# example()
# static_vs_dynamic()
# plot_demo()
# influence_of_infection_order()
