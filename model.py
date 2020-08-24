'''
Reference:
    https://www.idmod.org/docs/emod/generic/model-si.html#sis-without-vital-dynamics
'''

import numpy as np
import matplotlib.pyplot as plt

from graph import *

class Model:
    def __init__(self, G):
        self.G = G

class ModelSIS(Model):
    def __init__(self, G, init_state, beta, lambd):
        '''

        :param G: Graph
        :param init_state: initial status of population - 0: Susceptible 1: Infectious
        :param beta: transmission rate
        :param lambd: recovery rate
        '''
        assert len(G) == len(init_state)
        super(ModelSIS, self).__init__(G)
        self.beta = beta
        self.lambd = lambd
        self.state = init_state

    def evolve(self):
        new_state = np.empty((len(self.state)), dtype=int)
        for i in range(len(self.state)):
            if self.state[i] == 1:
                if np.random.rand() < self.lambd:
                    new_state[i] = 0
                else:
                    new_state[i] = 1
            else:
                count = ((self.state == 1) & (self.G.A[i] == True)).sum()
                if np.random.rand() < self.beta * count / len(self.state):
                    new_state[i] = 1
                else:
                    new_state[i] = 0
        self.state = new_state

    def drop_node(self, idx):
        self.A[idx, :] = False
        self.A[:, idx] = False

    def drop_edge(self, idx1, idx2):
        self.A[idx1, idx2] = False
        self.A[idx2, idx1] = False



def main():
    #G = GraphStochasticBlock(community_sizes=[10,20,30,40], edge_probabilities=np.ones((4,4))*0.1 + np.eye(4)*0.8)
    G = GraphFullyConnected(1000)
    m = ModelSIS(G, init_state=np.random.rand(len(G))<0.1, beta=0.05, lambd=0.02)
    n_infections = []
    for i in range(1000):
        n_infections.append(m.state.sum())
        m.evolve()
    plt.plot(n_infections)

if __name__ == '__main__':
    main()