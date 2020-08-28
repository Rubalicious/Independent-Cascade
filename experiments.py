from network_propagation import SquareLattice
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

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
    '''
    n = 15
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
        avg = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        return avg, std

    shuffle_avg, shuffle_std = stats(shuffle=1)
    ordered_avg, ordered_std = stats(shuffle=0)
    print(max(shuffle_std))

    # Error metrics
    abs_err_avg = np.abs(shuffle_avg-ordered_avg)
    rel_err_avg = np.abs(shuffle_avg-ordered_avg)/shuffle_avg
    mod_err_avg = np.abs(shuffle_avg-ordered_avg)**2/(shuffle_avg*ordered_avg)

    abs_err_std = np.abs(shuffle_std-ordered_std)
    rel_err_std = np.abs(shuffle_std-ordered_std)/shuffle_std
    mod_err_std = np.abs(shuffle_std-ordered_std)**2/(shuffle_std*ordered_std)

    # shuffle plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    img = np.reshape(shuffle_avg,(n,n+1))
    im1 = ax1.imshow(img, cmap='hot')
    ax1.set_title('average')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


    img = np.reshape(shuffle_std,(n,n+1))
    im2 = ax2.imshow(img, cmap='hot')
    ax2.set_title('standard deviation')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    fig.suptitle('active edges with shuffle')
    plt.savefig('active edges with shuffle')
    plt.show()

    # ordered plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    img = np.reshape(ordered_avg,(n,n+1))
    im1 = ax1.imshow(img, cmap='hot')
    ax1.set_title('average')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


    img = np.reshape(ordered_std,(n,n+1))
    im2 = ax2.imshow(img, cmap='hot')
    ax2.set_title('standard deviation')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    fig.suptitle('active edges without shuffle')
    plt.savefig('active edges without shuffle')
    plt.show()


    # absolute error plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    img = np.reshape(abs_err_avg,(n,n+1))
    im1 = ax1.imshow(img, cmap='hot')
    ax1.set_title('average')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


    img = np.reshape(abs_err_std,(n,n+1))
    im2 = ax2.imshow(img, cmap='hot')
    ax2.set_title('standard deviation')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    fig.suptitle('absolute error plots')
    plt.savefig('absolute error plots')
    plt.show()

    # relative error plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    img = np.reshape(rel_err_avg,(n,n+1))
    im1 = ax1.imshow(img, cmap='hot')
    ax1.set_title('average')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


    img = np.reshape(rel_err_std,(n,n+1))
    im2 = ax2.imshow(img, cmap='hot')
    ax2.set_title('standard deviation')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    fig.suptitle('relative error plots')
    plt.savefig('relative error plots')
    plt.show()

    # modified error plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    img = np.reshape(mod_err_avg,(n,n+1))
    im1 = ax1.imshow(img, cmap='hot')
    ax1.set_title('average')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")


    img = np.reshape(mod_err_std,(n,n+1))
    im2 = ax2.imshow(img, cmap='hot')
    ax2.set_title('standard deviation')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    fig.suptitle('modified error plots')
    plt.savefig('modified error plots')
    plt.show()



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
    n=26
    sq = SquareLattice(n)
    sq.dynamic_process(shuffle=1, view=1)

# example()
# static_vs_dynamic()
# plot_demo()
influence_of_infection_order()
