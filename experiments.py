from network_propagation import SquareLattice
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def static_vs_dynamic():
    n = 13
    sq = SquareLattice(n)

    # dynamic process:
    def dynamic_process(sq, shuffle=0):
        sq.infect([90])
        sq.view_current_state()
        A = sq.sample_active_edges()
        sq.view_active_edges(A)
        while not sq.is_totally_infected():
            sq.update(A, shuffle)
        v = sq.build_state_vector()/2
        sq.view_current_state()
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

    A,v = dynamic_process(sq,1)
    # A,v = static_process_exp(sq)
    N = 1
    for i in range(N):
        A,w = dynamic_process(sq, 1)
        # A,w = static_process_exp(sq)
        sys.stdout.write('\r')
        sys.stdout.write('{}% complete'.format(100*float(i)/N))
        sys.stdout.flush()
        v+=w
    # print(v/N)
    v = np.array(v)
    v = np.reshape(v,(n,n+1))
    plt.imshow(v/N, cmap='hot')
    plt.colorbar()
    # plt.plot(range(sq.n*sq.m), v/N)
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




main()
