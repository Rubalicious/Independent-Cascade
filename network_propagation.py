import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import os.path

class SquareLattice(object):
    """docstring for SquareLattice."""

    def __init__(self, n):
        super(SquareLattice, self).__init__()
        # dimensions of square lattice
        self.n = n
        self.m = n+1

        # create square lattice and collect positions of nodes
        if os.path.exists('test.edgelist'):
            self.G = self.load_graph()
        else:
            self.G = self.square_lattice(self.n,self.m)
            self.initialize_edge_weights()
            self.save_graph()

        # defines node positions
        self.pos = self.positions(self.n, self.m)

        # node colors need to reflect state: S (blue), I (red), R (green/black)
        self.node_color = ['b']*self.n*self.m
        # edge color may change to red to reflect active edges
        self.edge_color = ['k']*len(self.G.edges)

        self.infect()
        self.view_current_state()

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
            # self.G.nodes[node]['infectious'] = np.random.randint(0,5)
            self.G.nodes[node]['infectious'] = np.random.poisson(4.0)
            # susceptible: flag == 0, infected: flag == 1, recovered: flag == 2
            self.G.nodes[node]['flag'] = 0
            out_degree = len(list(self.G.neighbors(node)))
            # generate random numbers
            theta = np.random.uniform(0,1,(out_degree))
            i = 0
            for suc in list(self.G.neighbors(node)):
                # assign to each edge of a node's successors
                self.G[node][suc]['theta'] = np.round(theta[i], 3)
                self.G[node][suc]['active'] = 0
                i+=1

    def view_current_state(self,view_edge_weights=False):
        '''
            plots current state of network
        '''
        labels = nx.get_edge_attributes(self.G, 'theta')
        if view_edge_weights:
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=labels)
        # nx.set_node_attributes(self.G, 'pos', self.pos)
        nx.draw(self.G, self.pos,
            node_color=self.node_color,
            edge_color=self.edge_color,
            width=4
        )
        # nx.draw_networkx(self.G, self.pos, node_color=self.node_color)
        plt.title('IC_initial_condition')
        plt.savefig('IC_initial_condition')
        plt.show()

    def infect(self):
        '''
            choose one person to be infected and assign flags to each node
        '''
        # choose infected nodes here
        infected = [90]

        for node in self.G.nodes:
            if node in infected:
                self.node_color[node] = 'r'
                self.G.nodes[node]['flag'] = 1
            else:
                self.G.nodes[node]['flag'] = 0

    def build_state_vector(self):
        '''This outputs a vector of flags defining whether each
        node is infected or not'''
        N = self.n*self.m
        v = np.zeros((N,1))
        for i in self.G.nodes:
            if self.G.nodes[i]['flag'] == 1:
                v[i] = 1
            if self.G.nodes[i]['flag'] == 2:
                v[i] = 2
        return v


    def IC_step(self, A=None):
        '''
            Simulate one step of the Independent Cascade Model
            Can give a set of Active edges
        '''
        v = self.build_state_vector()
        infected = [idx for idx,val in enumerate(v) if val == 1]
        recovered = [idx for idx,val in enumerate(v) if val == 2]
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
                        self.G.nodes[nei]['flag'] = 1
                else:
                    # generate random number
                    rand = np.random.uniform(0,1)
                    if rand > self.G[inf][nei]['theta']:
                        # infect neighbor
                        self.G.nodes[nei]['flag'] = 1
            # COME BACK TO: need to keep track of how long each
            # infected node has been infectious for
            # this is the area of modification/generalization
            # then set infected node to recovered
            self.G.nodes[inf]['flag'] = 2

    def update(self, A=None):
        '''
            updates the graph, changes node colors, plots current state
        '''
        self.IC_step(A)
        # update colors
        for node in self.G.nodes:
            if self.G.nodes[node]['flag'] == 1:
                self.node_color[node] = 'r'
            if self.G.nodes[node]['flag'] == 2:
                self.node_color[node] = 'k'

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
        nx.draw(self.G, self.pos,
            node_color=self.node_color,
            edge_color=self.edge_color,
            width=4
        )
        plt.show()

    def static_process(self):
        '''
            make each node reachable to an infected node through active edges infected
        '''
        v = self.build_state_vector()
        infected = [idx for idx,val in enumerate(v) if val == 1]
        A = self.sample_active_edges()
        self.view_active_edges()

        # create temporary subgraph of active edges only
        Temp = nx.Graph()
        for node in self.G:
            Temp.add_node(node)
        for i,j in A:
            Temp.add_edge(i,j)

        # infect nodes that can be reached from the seed set through active edges
        for inf in infected:
            for node in Temp:
                if nx.has_path(Temp, inf, node):
                    self.node_color[node] = 'k'




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
        weights = [ G[sus][nei]['theta'] for nei in G.neighbors(sus) if G.nodes[nei]['flag'] == 1 ]
        tot = np.sum(weights)
        cum_weights[sus] = tot
        thresholds[sus] = G.nodes[sus]['tau']
        if tot > G.nodes[sus]['tau']:
            G.nodes[sus]['flag'] = 1
    # check if all remaining susceptibles have higher
    # threshold than the sum total of infected neighbor weights
    if (cum_weights <= thresholds).all(): terminate = 1
    return G, terminate



def demonstrate_active_edges():
    global G, edge_color
    # G,N = build_lattice_from(theta)
    labels = nx.get_edge_attributes(G,'theta')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    nx.draw_networkx(G, pos, node_color=node_color)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=3)
    plt.savefig('inactive_edges')
    plt.show()

    A = sample_active_edges()
    nx.draw_networkx(G, pos, node_color=node_color)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=3)
    plt.savefig('active_edges')
    plt.show()

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

def main():
    n = 13
    sq = SquareLattice(n)

    A = sq.sample_active_edges()
    print(A)
    sq.view_active_edges(A)

    while not sq.is_totally_infected():
        sq.update(A)
    #     plt.title('IC_iter_{}'.format(i))
    #     plt.savefig('IC_iter_{}'.format(i))

    # view active edges
    # sq.sample_active_edges()
    # sq.static_process()
    # sq.view_current_state()


main()
# demonstrate_active_edges(G)
