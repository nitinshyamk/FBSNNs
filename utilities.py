# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:59:43 2020

@author: nitin
"""

import numpy as np
import scipy.io as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


class CartPoleModel(object):
    ''' Creates an object to model cartpole dynamics with specified parameters
        Inputs:
            m = mass of pendulum (kg)
            M = Mass of cart (kg)
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
            A: K x D matrix where A[i, :, :] = a(x), and x is the state of
                i^th trajectory (x = X[i, :]). a(x) is a D  dimensional 
                array (technically D x 1 matrix). 
            B: K x D matrix where B[i, :, :] = b(x), and x is the state of
                i^th trajectory (x = X[i, :]). b(x) is a D  dimensional 
                array (technically D x 1 matrix). 
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
        
        x2 = X[:, 1]; x3 = X[:, 2]; x4 = X[:, 3];
        
        Ax1 = x2;
        
        Axdenom = np.ones(K) - r * np.square(np.cos(x3))
        
        Ax2_a = r * np.multiply(np.sin(2 * x3), g * np.ones(K) - r * L * np.multiply(np.square(x4), np.cos(x3)))
        Ax2_b = r * L *  np.multiply(np.square(x4), np.sin(x3))
        Ax2 = np.divide(Ax2_a, 2 * Axdenom) - Ax2_b;
        
        Ax3 = x4
        
        Ax4_a = g * np.sin(x3) / L - np.multiply(np.square(x4), np.sin(2 * x3)) * r / 2
        Ax4 = np.divide(Ax4_a, Axdenom);
        
        Bx1 = np.zeros(K);
        Bx2 = np.divide(r * r * np.square(np.cos(x3)), Axdenom) / m + np.ones(K) * r / m;
        Bx3 = np.zeros(K);
        Bx4 = np.multiply(np.divide(r * np.ones(K), m * L * Axdenom), np.cos(x3))
        
        rs = lambda a : a.reshape(-1, 1);
        
        A = np.concatenate((rs(Ax1), rs(Ax2), rs(Ax3), rs(Ax4)), axis = 1)
        B = np.concatenate((rs(Bx1), rs(Bx2), rs(Bx3), rs(Bx4)), axis = 1)
        
        return (A, B)

class PhasePortrait(object):
    @staticmethod
    def thetaDoubleDot(cartpole, controlValue = None):
        cp = cartpole
        tsz = 100; tdotsz = 200
        t, tdot = np.meshgrid(np.linspace(-1*np.pi, np.pi, tsz), np.linspace(- cp.g * cp.L , cp.g * cp.L, tdotsz))
        
        totalsz = tsz * tdotsz 
        allpoints = np.concatenate((
            np.zeros((totalsz, 1)), 
            np.zeros((totalsz, 1)), 
            np.reshape(t, (totalsz, 1)), 
            np.reshape(tdot, (totalsz, 1))), 
            axis = 1);
        
        if (controlValue == None):
            A, B = cp.f(allpoints);
            tddot = np.reshape(A[:, 3], (tdotsz, tsz));            
            data = dict(t = t, tdot = tdot, tddot = tddot)
            return data
            
'''
Runs through cart pole dynamics simulator over a few parameter settings and 
outputs the resulting phase plots to test
'''    
        
if __name__ == "__main__":
    paramcollection = [(1, 1, 1), (1, 10, 1), (1, 100, 0.5)]
    
    for (m, M, L) in paramcollection:
        cp = CartPoleModel(m, M, L)
        fname = "data_{}_{}_{}.mat".format(m, M, L)
        data = PhasePortrait.thetaDoubleDot(cp)
        sp.savemat(fname, data)


        
        
        
        
        
        

        