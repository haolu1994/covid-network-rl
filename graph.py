'''
Reference:
    https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2005.0051
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Graph(object):
    ''' A Placeholder for all graphs
    '''
    def __init__(self, A=None):
        self.A = A

    def __len__(self):
        return self.A.shape[0]

    @property
    def n_nodes(self):
        return len(self)

    @property
    def n_edges(self):
        return self.A.sum()

    def plot_adj_mat(self):
        sns.heatmap(self.A, cmap="OrRd")

    def _reset_diag(self):
        ''' Reset diagonal to 0
        '''
        for i in range(self.A.shape[0]):
            self.A[i, i] = 0

    def _reflect(self, lower=True):
        for i in range(len(self)):
            for j in range(i):
                if lower:
                    self.A[i, j] = self.A[j, i]
                else:
                    self.A[j, i] = self.A[i, j]

    def _check_adj_mat(self):
        assert self.A.shape[0] == self.A.shape[1]
        for i in range(self.A.shape[0]):
            for j in range(i):
                assert self.A[i, j] == self.A[j, i]
            assert self.A[i, i] == 0

class GraphFullyConnected(Graph):
    def __init__(self, n):
        assert n >= 1
        self.A = np.ones((n,n), dtype=bool)
        for i in range(n):
            self.A[i,i] = False

class GraphRandom(Graph):
    def __init__(self, n, p):
        '''

        :param n: Number of nodes
        :param p: Probability of existence of each edge
        '''
        assert n >= 1
        assert (p >= 0) and (p <= 1)
        self.generating_p = p
        self.A = np.random.rand(n, n) < p
        self._reset_diag()
        self._reflect(lower=True)

class GraphLattices(Graph):
    def __init__(self, height, width):
        '''

        :param width: width of the grid
        :param height: height of the grid
        '''
        assert width >= 1
        assert height >= 1
        self.grid_width = width
        self.grid_height = height
        n = width * height

        self.A = np.zeros((n, n), dtype=bool)
        for i in range(height):
            for j in range(width):
                if j != width-1:
                    self.A[self.loc2idx(i, j), self.loc2idx(i, j+1)] = True
                    self.A[self.loc2idx(i, j+1), self.loc2idx(i, j)] = True
                if i != height-1:
                    self.A[self.loc2idx(i, j), self.loc2idx(i+1, j)] = True
                    self.A[self.loc2idx(i+1, j), self.loc2idx(i, j)] = True

        self._check_adj_mat()

    def loc2idx(self, i, j):
        assert i < self.grid_height and j < self.grid_width
        return i * self.grid_width + j

    def idx2loc(self, idx):
        return idx // self.grid_width, idx % self.grid_width

class GraphSmallWorld(Graph):
    def __init__(self, n, phi):
        assert n >= 1
        assert (phi >= 0) and (phi <= 1)
        self.generating_phi = phi
        self.A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            self.A[i, (i + 1) % n] = True
            self.A[(i + 1) % n, i] = True
        self.A |= np.random.rand(n, n) > phi
        self._reset_diag()
        self._reflect(lower=True)

        self._check_adj_mat()

class GraphSpatial(Graph):
    pass

class GraphStochasticBlock(Graph):
    def __init__(self, community_sizes, edge_probabilities):
        ''' See https://en.wikipedia.org/wiki/Stochastic_block_model

        :param community_sizes: sizeof(C1), ..., sizeof(Cr)
        :param edge_probabilities: Pij = prob of u \in Ci and v \in Cj are connected by an edge
        '''
        community_sizes, edge_probabilities = np.array(community_sizes), np.array(edge_probabilities)
        assert len(community_sizes) >= 1 and np.all(community_sizes >= 1)
        r = len(community_sizes)
        n = community_sizes.sum()
        assert edge_probabilities.shape == (r, r)
        self.community_sizes = community_sizes
        self.edge_probabilities = edge_probabilities
        self._size_cumsum = community_sizes.cumsum()
        self.idx2community = np.empty((n,), dtype=int)
        for c in range(r):
            self.idx2community[self.community2idxes(c)] = c

        self.A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i):
                self.A[i, j] = np.random.rand() < edge_probabilities[self.idx2community[i], self.idx2community[j]]
                self.A[j, i] = self.A[i, j]

        self._check_adj_mat()

    def community2idxes(self, c):
        right = self._size_cumsum[c]
        left = self._size_cumsum[c-1] if c >= 1 else 0
        return np.array(range(left, right))






