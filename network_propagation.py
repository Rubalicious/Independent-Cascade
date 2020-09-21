import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import os.path
import random
import sys

class SquareLattice(object):
    """docstring for SquareLattice."""

    def __init__(self, n):
        super(SquareLattice, self).__init__()
        # dimensions of square lattice
        self.n = n
        self.m = n+1
        # number of nodes
        self.N = n*(n+1)

        # create square lattice and collect positions of nodes
        # if os.path.exists('squarelattice.graphml'):
        #     self.G = self.load_graph()
        # else:
        self.G = self.square_lattice(self.n,self.m)
        self.initialize_edge_weights()
        # self.save_graph()

        # defines node positions
        self.pos = self.positions(self.n, self.m)

        # node colors need to reflect state: S (blue), I (red), R (green/black)
        self.node_color = ['b']*self.n*self.m
        # edge color may change to red to reflect active edges
        self.edge_color = ['k']*len(self.G.edges)

        # list of originally infected nodes
        mid_seed = np.floor(self.N/2) if self.N%2 == 0 else np.floor(self.N/2 + self.n/2)
        self.seed = [mid_seed]

        # self.infect()
        # self.view_current_state()

    def save_edge_list(self):
        nx.write_weighted_edgelist(self.G, 'test.edgelist')

    def load_edge_list(self):
        return nx.read_edgelist('test.edgelist')

    def save_graph(self):
        nx.write_graphml(self.G, 'squarelattice.graphml')

    def load_graph(self):
        return nx.read_graphml('squarelattice.graphml')

    # generate square lattice
    def square_lattice(self, n=4,m=5):
        '''
            Creates a square lattice graph of n x m nodes
        '''
        L = 1.0; M = 1.0;
        G = nx.Graph()
        c = 0
        # adding nodes in square lattice
        for i in range(n):
            for j in range(m):
                G.add_node(c)
                G.node[c]['x'] = L*float(i)/n
                G.node[c]['y'] = M*float(j)/m
                c+=1
        # add edges between nodes
        c = 0
        N = n*m
        for i in range(n):
            for j in range(m):
                if c+m < N:
                    G.add_edge(c, c+m)
                if c%m != n:
                    G.add_edge(c, (c+1)%N)
                c+=1
        return G

    def positions(self, n=4, m=5):
        L = 1.0; M=1.0
        pos = {}
        c = 0
        # adding nodes in square lattice
        for i in range(n):
            for j in range(m):
                pos[c] = [L*float(i)/n, M*float(j)/m]
                c+=1
        return pos

    def initialize_edge_weights(self):
        '''
            associate weights to each edge.
            these are probabilities that one node infects the next
        '''
        for node in self.G.nodes:
            # each node is infectious for some number of days
            # assign number of days infectious for and a count to keep track
            # self.G.nodes[node]['infectious'] = [np.random.randint(0,5), 0]
            self.G.nodes[node]['infectious'] = [0, 0]
            # self.G.nodes[node]['infectious'] = [np.random.poisson(4.0), 0]
            # susceptible: state == 0, infected: state == 1, recovered: state == 2
            self.G.nodes[node]['state'] = 0
            out_degree = len(list(self.G.neighbors(node)))
            # generate random numbers
            theta = np.random.uniform(0,1,(out_degree))
            i = 0
            for suc in list(self.G.neighbors(node)):
                # assign to each edge of a node's successors
                self.G[node][suc]['theta'] = np.round(theta[i], 3)
                assert(self.G[node][suc]['theta'] == self.G[suc][node]['theta'])
                # may need to be symmetric
                # self.G[suc][node]['theta'] = np.round(theta[i], 3)
                # self.G[node][suc]['active'] = 0
                i+=1


    def view_current_state(self,view_edge_weights=False):
        '''
            plots current state of network
        '''
        labels = nx.get_edge_attributes(self.G, 'theta')

        # to view weights associated for each edge
        if view_edge_weights:
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=labels)

        nx.draw(self.G, self.pos,
            node_color=self.node_color,
            edge_color=self.edge_color,
            width=2,
            node_size=90
        )

        plt.title('IC current state')
        # plt.savefig('IC current state')
        plt.show()

    def plot_demo(self, n):
        fig, axs = plt.subplots(n,n)

        self.infect([15])

        A = self.sample_active_edges()

        c = 0
        for i in range(n):
            for j in range(n):
                nx.draw(self.G, self.pos,
                    node_color=self.node_color,
                    edge_color=self.edge_color,
                    width=2,
                    node_size=50,
                    ax = axs[i,j]
                )
                axs[i,j].set_title('t = {}'.format(c))
                self.IC_step(A,shuffle=1)
                c+=1
        plt.show()



    def infect(self, infected=None):
        '''
            choose one person to be infected and assign states to each node
        '''
        # choose infected nodes here
        if infected is None:
            infected = self.seed
        for node in self.G.nodes:
            if node in infected:
                self.node_color[node] = 'r'
                self.G.nodes[node]['state'] = 1
            else:
                self.G.nodes[node]['state'] = 0

    def build_state_vector(self):
        '''This outputs a vector of states defining whether each
        node is infected or not'''
        N = self.n*self.m
        v = np.zeros((N,1))
        for i in self.G.nodes:
            if self.G.nodes[i]['state'] == 1:
                v[i] = 1
            if self.G.nodes[i]['state'] == 2:
                v[i] = 2
        return v


    def IC_step(self, A=None, shuffle=0):
        '''
            Simulate one step of the Independent Cascade Model
            Can give a set of Active edges
            Can shuffle order of iterating through infected nodes
        '''
        v = self.build_state_vector()
        infected = [idx for idx,val in enumerate(v) if val == 1]
        recovered = [idx for idx,val in enumerate(v) if val == 2]

        current_state = self.node_color
        if shuffle == 1:
            random.shuffle(infected)
        # for each infected node
        for inf in infected:
            # and for each neighbor of the infected node
            for nei in self.G.neighbors(inf):
                # skip node if neighbor is recovered
                if nei in recovered:    continue

                # infect neighbors according to active edges
                if A is not None:
                    l = [inf,nei]
                    l.sort()
                    l = tuple(l)
                    if l in A:
                        self.G.nodes[nei]['state'] = 1
                        self.node_color[nei] = 'r'
                        # if current_state != self.node_color:
                        # self.view_current_state()
                else:
                    # generate random number
                    rand = np.random.uniform(0,1)
                    if rand > self.G[inf][nei]['theta']:
                        # infect neighbor
                        self.G.nodes[nei]['state'] = 1
                        self.node_color[nei] = 'r'
                #     self.view_current_state()
            # extract infectiousness of node
            infectious = self.G.nodes[inf]['infectious']
            if infectious[1] == infectious[0]:
                # node recovers
                self.G.nodes[inf]['state'] = 2
                self.node_color[inf] = 'k'
            else:
                # it is infectious for one more time step
                self.G.nodes[inf]['infectious'][1] += 1


    def update(self, A=None, shuffle=1):
        '''
            updates the graph, changes node colors, plots current state
        '''
        self.IC_step(A, shuffle)
        self.view_current_state()

    def is_totally_infected(self):
        '''
            determines if IC process is finished
        '''
        v = self.build_state_vector()
        if (v == 1).any():  return False
        return True

    def sample_active_edges(self):
        '''
            samples active edges and returns a list of them
            NEEDS TO BE ADAPTED FOR NODES INFECTIOUS FOR MULTIPLE TIME STEPS
            returns a list of tuples
        '''
        A = []
        c = 0
        for i,j in self.G.edges:
            rand = np.random.uniform(0,1)
            if rand > self.G[i][j]['theta']:
                # G[i][j]['active'] = 1
                A.append((i,j))
                self.edge_color[c] = 'r'
            c+=1
        return A # returning a set of active edges

    def view_active_edges(self, A=None):
        '''
            plots the active edges
        '''
        if A is None:
            A = self.sample_active_edges()
        c = 0
        for edge in self.G.edges:
            if edge in A:
                self.edge_color[c] = 'r'
            c+=1
        nx.draw(self.G, self.pos,
            node_color=self.node_color,
            edge_color=self.edge_color,
            width=4
        )
        plt.show()

    def static_process(self, A=None):
        '''
            make each node reachable to an infected node through active edges infected
        '''
        v = self.build_state_vector()
        infected = [idx for idx,val in enumerate(v) if val == 1]
        if A is None:
            A = self.sample_active_edges()
        # self.view_active_edges()

        # create temporary subgraph of active edges only
        Temp = nx.Graph()
        for node in self.G:
            Temp.add_node(node)
        for i,j in A:
            Temp.add_edge(i,j)

        # infect nodes that can be reached from the seed set through active edges
        for inf in infected:
            for node in self.G.nodes:
                if nx.has_path(Temp, inf, node):
                    self.node_color[node] = 'k'
                    self.G.nodes[node]['state'] = 2

    def reset(self):
        '''
            resets the the graph to all susceptible without changing weights
            this resets all active edges too
        '''
        self.node_color = ['b']*self.n*self.m
        self.edge_color = ['k']*len(self.G.edges)
        for node in self.G.nodes:
            self.G.nodes[node]['state'] = 0
            self.G.nodes[node]['infectious'][1] = 0

    def collect_final_configuration(self):
        '''
            collect all recovered nodes
        '''
        pass

    def get_active_edge_weights(self):
        '''
            extract the active edge weights in the form of an adjacency matrix
            whose values (g)_ab is the weight for probability of activation.
        '''
        n = len(self.G.nodes)
        g = np.zeros((n,n))
        thetas = np.zeros((n,n))
        for a,b in self.G.edges:
                theta = self.G[a][b]['theta']
                # extract random weights
                thetas[a,b] = theta
                thetas[b,a] = theta

                # convert to underlying parameter
                g[a,b] = -np.log(1/theta - 1)
                # make it symmetric (may relax/omit later)
                g[b,a] = g[a,b]
        return g, thetas

    def dynamic_process(self, shuffle=0, view=0):
        self.infect()
        sigma_in = self.build_state_vector()/2
        if view == 1: self.view_current_state()
        while not self.is_totally_infected():
            self.IC_step(shuffle=shuffle)
            if view == 1: self.view_current_state()
        sigma = self.build_state_vector()/2
        self.reset()
        return sigma_in, sigma


#========================================================
#
#       AUXILARY FUNCTIONS
#
#========================================================

def build_lattice_from(wei_adj):
    G = nx.Graph()
    # N is the number of nodes
    N,_ = np.shape(wei_adj)

    # add nodes
    for i in range(N):
        G.add_node(i)

    # add edges
    for i in range(N):
        for j in range(N):
            if wei_adj[i][j] != 0:
                G.add_edge(i,j)
                G[i][j]['theta'] = wei_adj[i][j]
    return G, N

# build adjacency matrix
def build_adj(G):
    n = len(list(G.nodes))
    adj = np.zeros((n,n))
    for i in G.nodes:
        for j in G.nodes:
            if (i,j) in G.edges:
                adj[i,j] = 1
                adj[j,i] = 1
    return adj

# build weighted adjacency matrix
def build_weighted_adj(G):
    n = len(list(G.nodes))
    wei_adj = np.zeros((n,n))
    for i in G.nodes:
        for j in G.nodes:
            if (i,j) in G.edges:
                wei_adj[i,j] = G[i][j]['theta']
                wei_adj[j,i] = G[i][j]['theta']
    return wei_adj

# Probably won't use this anymore
# Linear Threshold Model
def LT_step():
    global G
    v = build_state_vector()
    susceptible = [idx for idx,val in enumerate(v) if val == 0]
    terminate = 0
    # keep track of cumulative weights of infected neighbors for each person
    cum_weights = np.zeros((len(v),1))
    thresholds = np.zeros((len(v),1))
    for sus in susceptible:
        # collect weights of infected neighbors of sus
        weights = [ G[sus][nei]['theta'] for nei in G.neighbors(sus) if G.nodes[nei]['state'] == 1 ]
        tot = np.sum(weights)
        cum_weights[sus] = tot
        thresholds[sus] = G.nodes[sus]['tau']
        if tot > G.nodes[sus]['tau']:
            G.nodes[sus]['state'] = 1
    # check if all remaining susceptibles have higher
    # threshold than the sum total of infected neighbor weights
    if (cum_weights <= thresholds).all(): terminate = 1
    return G, terminate


def number_of_reachable_nodes(x,A):
    # count the number of nodes that can be reached through A
    # from node x
    # i.e. count number of edges in A contianing x
    return len([1 for a in A if x in a])


def compute_infection_kernel(G):
    # need to compute generate two sets of seed nodes A and A'
    A1 = sample_active_edges()
    A2 = sample_active_edges()
    # need to compute number of reachable nodes r(x,A) from x through A
    r = lambda x,A: len([1 for a in A if x in a])
    # define the correlation coefficients
    Q = lambda e1, e2: 1.0/3 if e1 == e2 else 1.0/6

    K = 0
    return A1,A2
