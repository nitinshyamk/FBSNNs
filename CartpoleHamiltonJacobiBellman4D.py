"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow.compat.v1 as tf
from FBSNNs import FBSNN
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from utilities import CartPoleModel

class CartpoleHamiltonJacobiBellman4D(FBSNN):
    '''
    Creates and initializes a FBSNN for the Cartpole 4D HJB PDE system.
    Inputs:
        Cartpole: Cartpole dynamics model (already parametrized)
        Q: Rolling cost matrix for input state dynamics
        R: Rolling cost matrix (1 x 1 constant) for control 
        Qt: Terminal cost matrix for final state
        snoise: Scale of stochastic noise in cartpole model (sigma)
            
        Xi: initial state 
        T: End Time
        N: Number of timesteps (dt = T / N)
        K: Number of trajectories to simulate in a batch
        D: Dimension of system
        layers: list of dimensions corresponding to layer dimensions
            ex: [l0, l1, l2, l3] creates a 3 layer neural network
            mapping from dimension l0 -> l1 -> l2 -> l3
            
            layers[0] must equal D + 1 (time) and layers[n - 1] must equal 1
    '''
    def __init__(self, Cartpole, Q, R, Qt, snoise, Xi, T, N, K, D, layers):
        self.cartpole = Cartpole
        self.Q = Q
        self.R = R
        self.Qt = Qt
        self.snoise = snoise
        super().__init__(Xi, T, K, N, D, layers)
        
    
    ### REQUIRED IMPLEMENTATION ###
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        const = tf.constant
        
        phi1 = const(0.5) * self.quadraticForm(X, self.Q)
        A, B = self.cartpole.f(X)
        BtXgradV = self.BtXgradV(B, Z);
        phi2 = const(0.5) * tf.square(BtXgradV) / const(self.R * 1.0)
        
        return phi1 + phi2
        
    
    def g_tf(self, X): # M x D
        return self.quadraticForm(X, self.Qt); # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
    
        A, B = self.cartpole.f(X)
        matmul, multiply, rowsum = self.getTFUtils()
        
        uinput = tf.constant(1.0 / self.R) * self.BtXgradV(B, Z)
        inputs = tf.repeat(tf.reshape(uinput, (-1, 1)), repeats = 4, axis = 1)
        
        return A - multiply(inputs, B) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.constant(self.snoise) * tf.matrix_diag(tf.ones([M,D]))
    
    #### HELPER METHODS
    
    def BtXgradV(self, Bx, Vgrad): # M x D, M x D
        matmul, multiply, rowsum = self.getTFUtils()
        return rowsum(multiply(Bx, Vgrad))
    
    def quadraticForm(self, X, PSD): # M x D, D x D
        matmul, multiply, rowsum = self.getTFUtils()
        return rowsum(multiply(matmul(X, PSD), X)) # M x 
    
    def getTFUtils(self):
        matmul = tf.linalg.matmul
        multiply = tf.math.multiply
        rowsum = lambda x : tf.math.reduce_sum(x, axis = 1);        
        return (matmul, multiply, rowsum)
    ###########################################################################


if __name__ == "__main__":
    tf.disable_eager_execution()
    
    
    # training and system configuration
    T = 1.0
    K = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 4 # number of dimensions
    layers = [D+1] + 4*[32] + [1]
    
    # cartpole parameters
    m = 1
    M = 1
    L = 1
    Xi = np.zeros([1,D])
    
    cp = CartPoleModel(m, M, L);
    Q = np.identity(D);
    Qt = np.identity(D);
    snoise = 0.01
    R = 1
    
         
    # Training
    model = CartpoleHamiltonJacobiBellman4D(cp, Q, R, Qt, snoise, Xi, T, N, K, D, layers);
        
    model.train(N_Iter = 2*10**4, learning_rate=1e-3)
    model.train(N_Iter = 3*10**4, learning_rate=1e-5)
    model.train(N_Iter = 2*10**4, learning_rate=1e-6)
    
    
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    