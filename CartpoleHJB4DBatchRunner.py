# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 04:54:16 2020

@author: nitin
"""
import numpy as np
from CartPoleModel import *
from CartpoleHamiltonJacobiBellman4D import *

class CartpoleHJBParams:
    def __init__(self, T, K, N, D, Q, Qt, R, snoise, layers):
        self.T = T
        self.K = K
        self.N = N
        self.D = D
        self.Q = Q
        self.Qt = Q
        self.R = R
        self.snoise = snoise
        self.layers = layers
    
    def print(self):
        print("T: {}, N: {}, K: {}, D: {}, Q: {}, Qt: {}, R: {}, snoise: {}, layersz: {} ".format(self.T, self.N, self.K, self.D, self.Q, self.Qt, self.R, self.snoise, self.layers[2]))
        
if __name__ == "__main__":    
    tf.disable_eager_execution()

    # cartpole model
    m = 1
    M = 1
    L = 1
    cartpole = CartPoleModel(m, M, L)
    D = 4
    
    # trajectories
    K = 5000
    Xi = np.zeros([1,D])

    # timesteps
    N = 50
    T = 1.0
    
    # layers 
    multilayers = [([D+1] + 4*[sz] + [1]) for sz in [32, 64, 128, 256]]
    
    defaultlayer = [D+1] + 4*[64] + [1]
    
    # costs 
    Q = np.identity(D);
    Qt = np.identity(D);
    snoise = 0.01
    R = 1
    
    createparams = lambda layers : CartpoleHJBParams(T, K, N, D, Q, Qt, R, snoise, layers)
    params = [createparams(layers) for layers in multilayers]
    
    def train_meta(model):
        model.train(N_Iter = 2*10**3, learning_rate=1e-3)
        model.train(N_Iter = 3*10**3, learning_rate=1e-4)
        model.train(N_Iter = 3*10**3, learning_rate=1e-5)
        return model.train(N_Iter = 2*10**3, learning_rate=1e-6);
    
    losses = []
    for p in params:
        model = CartpoleHamiltonJacobiBellman4D(cartpole, p.Q, p.R, p.Qt, p.snoise, Xi, p.T, p.N, p.K, p.D, p.layers);
        losses.append(train_meta(model))


    for i in range(len(params)):
        params[i].print()
        print(losses[i])
        
        
    
    
    
    
    
