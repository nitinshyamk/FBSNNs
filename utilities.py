# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:59:43 2020

@author: nitin
"""

import numpy as np


class CartPoleDynamics(object):
    ''' Creates an object to model cartpole dynamics with specified parameters
        Inputs:
            M = Mass of cart (kg)
            m = mass of pendulum (kg)
            L = length of pendulum (m) 
    '''
    def __init__(self, m, M, L):
        m = 1.0 * m; M = 1.0 * M; L = 1.0 * L;
        self.m = m
        self.M = M
        self.L = L
        self.g = 9.80665
        
    '''f is the function governing the dynamics of the nonlinear cartpole 
        system. Here, f(x, u) is the model in the equation: 
            dx = f(x, u) * dt + C * dW_t
        and in the cartpole system, f(x) takes the specific form, semilinear 
        in the input u.
            f(x) = a(x) + B(x) u. 
            
        This function takes an input of state vectors for K trajectories and
        returns the tuple of matrices (A, B). Details follow:
        
        Inputs:
            X : K x D matrix of states corresponding to the D dimensional state 
                representations of K different trajectories
        Outputs:
            A: K x D x 1 tensor where A[i, :, :] = a(x), and x is the state of
                i^th trajectory (x = X[i, :]). a(x) is an D x 1 dimensional 
                matrix. 
            B: K x D x 1 where B[i, :, :] = b(x), and x is the state of
                i^th trajectory (x = X[i, :]). b(x) is an D x 1 dimensional 
                matrix. 
  '''
    def f(self, X):
        m = self.m
        M = self.M
        L = self.L
        g = self.g
        r = m / (m + M)
        K, D = X.shape
        # gravitational constant
        
        
        if (D != 4):
            raise Exception("incompatible cartpole dimensions")
        
        x1 = X[:, 1]; x2 = X[:, 2], x3 = X[:, 3], x4 = X[:, 4];
        
        Ax1 = x2;
        
        Axdenom = (np.ones(K, 1) - r * np.square(np.cos(x3)))
        
        Ax2_a = r * np.multiply(np.sin(2 * x3), g * np.ones(K, 1) - r * L * np.multiply(np.square(x4), np.cos(x3)))
        Ax2_b = r * L *  np.multiply(np.square(x4), np.sin(x3))
        Ax2 = np.divide(Ax2_a, 2 * Axdenom) - Ax2_b;
        
        Ax3 = x4
        
        Ax4_a = g * np.sin(x3) / L - np.multiply(np.square(x4), np.sin(2 * x3)) * r / 2
        Ax4 = np.divide(Ax4_a, Axdenom);
        
        Bx1 = np.zeros(K, 1);
        Bx2 = np.divide(r * r * np.square(np.cos(x3)), Axdenom) / m + np.ones(K, 1) * r / m;
        Bx3 = np.zeros(K, 1);
        Bx4 = np.multiply(np.divide(r * np.ones(K, 1), m * L * Axdenom), np.cos(x3))
        
        A = np.concatenate((Ax1, Ax2, Ax3, Ax4), axis = 0)
        B = np.concatenate((Bx1, Bx2, Bx3, Bx4), axis = 0)
        
        return (A, B)
        
        