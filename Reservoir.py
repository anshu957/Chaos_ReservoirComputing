'''
File: Reservoir.py
File Created: Sunday, 7th July 2019 5:26:25
Author: well-well-well
-------------------------------------------
Description:  Implementation of Reservoir
-------------------------------------------
'''

import numpy as np
import scipy
import networkx as nx


class Reservoir(object):
    '''
        Creates a Reservoir with weights and specific spectral radius
    '''

    def __init__(self, res_size, input_dim,
                 spectral_radius, leaky_rate, res_density, inputScaling_radius, input_density, random_seed=12837, beta=1e-03):
        super(Reservoir, self).__init__()

        self.input_dim = input_dim
        self.res_size = res_size

        self.res_density = res_density
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate
        self.inputScaling_radius = inputScaling_radius
        self.input_density = input_density
        self.random_seed = random_seed
        self.beta = beta

        self.W_out = None
        self.res = np.zeros((self.res_size, 1))

        self.W_res = self.createReservoir()
        self.W_in = self.getInputConnectionMatrix()

    # Class functions ...

    def createReservoir(self):
        '''
        Creates the reservoir (exluding input and output layers) matrix with given 
        spectral radius

        Returns: Reservoir encoded in the weighted adjacency matrix
        '''

        g = nx.erdos_renyi_graph(
            self.res_size, self.res_density, directed=False, seed=self.random_seed)

        # Assigning wieghts to the edges from a uniform distribition [-1:1]
        for u, v, d in g.edges(data=True):
            d['weight'] = np.random.uniform(-1, 1)

        # getting adjacency matrix for eigen value calculations
        adj_mat = scipy.sparse.csc_matrix.todense(nx.adj_matrix(g))

        # maximum eigen value representing spectral radius of the randomly constructed reservoir
        rhoW = max(abs(np.linalg.eig(adj_mat)[0]))

        # rescaling to make spectral radius = 0.8
        W_res = self.spectral_radius * adj_mat / rhoW

        return W_res

    def getInputConnectionMatrix(self):
        '''
        Constructs the input-to-reservoir connection matrix [W_in]
        which has dimensions (reservoir_size, 1+input_dimension)
        '''

        W_in = np.zeros((self.res_size, 1 + self.input_dim))

        for i in range(self.res_size):
            for j in range(1 + self.input_dim):
                ran_toss = np.random.random()
                if ran_toss < self.input_density:
                    W_in[i, j] = self.inputScaling_radius * \
                        (1.0 - 2.0 * np.random.random())

        return W_in

    def train(self, train_data, transience_len):
        '''
        Evolves the reservoir and train the weights of the output layer  
        connections using linear regression
        '''

        train_len = len(train_data) - 1
        train_mat = np.zeros(
            (1 + self.res_size + self.input_dim, train_len - transience_len))
        res = np.zeros((self.res_size, 1))
        out = np.zeros((self.res_size, self.input_dim))
        target_data = np.zeros((self.input_dim, train_len - transience_len))
        for i in range(train_len):
            U = np.concatenate(([1], train_data[i, :].flatten()), 0)

            # print(U.shape)
            self.res = (1 - self.leaky_rate) * self.res + self.leaky_rate * \
                np.tanh(np.dot(self.W_res, self.res) +
                        np.dot(self.W_in, U.reshape(U.size, 1)))

            # discarding reservoir steps for some transience length. This is the transience for reservoir
            if i >= transience_len:
                tmp = np.concatenate(
                    (U.reshape(U.size, 1), self.res.reshape(self.res.size, 1)), axis=0)
                train_mat[:, i - transience_len] = tmp.reshape(tmp.size)
                target_data[:, i -
                            transience_len] = np.transpose(train_data[i + 1, :])

        # Learining the weights using linear regression
        # ----------------------------
        self.W_out = np.dot(np.dot(target_data, np.transpose(train_mat)), np.linalg.inv(
            np.dot(train_mat, np.transpose(train_mat)) + self.beta * np.identity(1 + self.input_dim + self.res_size)))

        print('----------------------------')
        print('Training complete...')
        print('----------------------------')

    def generateFutureTrajectories(self, test_data):
        '''
         Generates future time series in "generative mode". In this mode, the
         reservoir acts a dynamical system feeding its output as the input for next time 

        Returns : [states, outputs]
            >> "states" has shape => (test_data_length, reservoir_size) and contains the state of all neurons in time.
            >> "outputs" has shape => (test_data_length, input_dimension) and contains the generated output state u(t+1) for u(t).
        '''

        Y = np.zeros((self.input_dim, 1))
        states = np.zeros((len(test_data), self.res_size))
        outputs = np.zeros((len(test_data), self.input_dim))
        U = np.concatenate(([1], test_data[0, :].flatten()), 0)
        for i in range(len(test_data)):

            self.res = ((1 - self.leaky_rate) * self.res +
                        self.leaky_rate * np.tanh(np.dot(self.W_res, self.res) + np.dot(self.W_in, U.reshape(U.size, 1))))

            tmp = np.concatenate(
                (U.reshape(U.size, 1), self.res.reshape(self.res.size, 1)), axis=0)
            Y = np.dot(self.W_out, tmp.reshape(tmp.size, 1))
            states[i, :] = self.res.reshape(self.res.size)
            outputs[i, :] = Y.reshape(Y.size)

            U = np.concatenate(([1], outputs[i, :].flatten()), 0)

        return states, outputs
