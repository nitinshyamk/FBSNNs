"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow.compat.v1 as tf
from FBSNNs import FBSNN
import matplotlib.pyplot as plt
from plotting import newfig, savefig

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
    def __init__(self, Cartpole, Q, R, Qt, snoise, Xi, T, N, K, D, layers, ):
        
        super().__init__(Xi, T, K, N, D, layers)
        self.cartpole = Cartpole
        self.Q = Q
        self.R = R
        self.Qt = Qt
        self.snoise = snoise
    
    ### REQUIRED IMPLEMENTATION ###
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        phi1 = 0.5 * np.matmul(np.matmul(X, self.Q), X)
        
        A, B = self.cartpole.f(X)
        BtXgradV = self.BtXgradV(B, Z);
        phi2 = 0.5 * np.square(BtXgradV) / self.R
        
        return phi1 + phi2
        
    
    def g_tf(self, X): # M x D
        return self.quadraticForm(X, self.Qt);

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        A, B = self.cartpole.f(X)
        
        uinput = (1.0 / self.R) * self.BtXgradV(B, Z)
        
        return A - B * uinput
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return self.snoise * np.identity(self.D)
    
    #### HELPER METHODS
    
    def BtXgradV(self, Bx, Vgrad): # M x D, M x D
        return np.sum(np.multiply(Bx, Vgrad), axis = 1)
    
    def quadraticForm(self, X, PSD): # M x D, D x D
        return np.sum(np.multiply(np.matmul(X, PSD), X), axis = 1);
    ###########################################################################


if __name__ == "__main__":
    tf.disable_eager_execution()
    M = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    
    layers = [D+1] + 4*[256] + [1]

    Xi = np.zeros([1,D])
    T = 1.0
         
    # Training
    model = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers)
        
    model.train(N_Iter = 2*10**4, learning_rate=1e-3)
    model.train(N_Iter = 3*10**4, learning_rate=1e-4)
    model.train(N_Iter = 3*10**4, learning_rate=1e-5)
    model.train(N_Iter = 2*10**4, learning_rate=1e-6)
    
    
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    
    def g(X): # MC x NC x D
        return np.log(0.5 + 0.5*np.sum(X**2, axis=2, keepdims=True)) # MC x N x 1
        
    def u_exact(t, X): # NC x 1, NC x D
        MC = 10**5
        NC = t.shape[0]
        
        W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
        
        return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W)),axis=0))
    
    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])
    
    Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))
    
    plt.figure()
    plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned $u(t,X_t)$')
    #plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
    plt.plot(t_test[0,:,0].T,Y_test[:,0].T,'r--',label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1,-1,0],Y_test_terminal[0:1,0],'ks',label='$Y_T = u(T,X_T)$')
    #plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
    plt.plot([0],Y_test[0,0],'ko',label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    plt.legend()
    
    # savefig('./figures/HJB_Apr18_50', crop = False)
    
    errors = np.sqrt((Y_test-Y_pred[0,:,:])**2/Y_test**2)
    
    plt.figure()
    plt.plot(t_test[0,:,0],errors,'b')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    # plt.legend()
    
    # savefig('./figures/HJB_Apr18_50_errors', crop = False)