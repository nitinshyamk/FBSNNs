# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:59:43 2020

@author: nitin
"""

import tensorflow.compat.v1 as tf

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
            X : K x D tensor of states corresponding to the D dimensional state 
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

        #tensorize constants
        rt = tf.constant(r)
        gt = tf.constant(g)
        mt = tf.constant(m)
        Lt = tf.constant(L)

        
        if (D != 4):
            raise Exception("incompatible cartpole dimensions")
        
        x2 = X[:, 1]; x3 = X[:, 2]; x4 = X[:, 3];
        
        Ax1 = x2;
        
        sm = tf.math.scalar_mul
        cos = tf.math.cos
        sin = tf.math.sin
        square = tf.math.square
        multiply = tf.math.multiply
        divide = tf.math.divide
        ones = lambda dim : tf.ones(dim, dtype = tf.float32)
        zeros = lambda dim : tf.zeros(dim, dtype = tf.float32)
        
        Axdenom = sm(Lt, sm(tf.constant(4.0 / 3.0), ones(K)) - sm(rt, square(cos(x3))))
        
        Ax4num = sm(gt, sin(x3)) - sm(tf.constant(0.5) * rt * Lt, multiply(square(x4), sin(sm(tf.constant(2.0), x3))))
        Ax4 = divide(Ax4num, Axdenom)
        
        Ax2_a = sm(rt * Lt, multiply(square(x4), sin(x3)))
        Ax2_b = sm(rt * Lt, cos(x3))
        Ax2 = Ax2_a - divide(Ax2_b, Axdenom)
        
        Ax1 = x2
        Ax3 = x4
        
        Bx1 = zeros(K);
        Bx2 = divide(sm(rt * rt * Lt, square(cos(x3))), sm(mt, Axdenom)) + sm(rt / mt, ones(K))
        Bx3 = zeros(K);
        Bx4 = divide(sm(rt, cos(x3)), sm(mt, Axdenom))
        
        rs = lambda a : tf.reshape(a, [-1, 1]); 
        
        A = tf.concat([rs(Ax1), rs(Ax2), rs(Ax3), rs(Ax4)], axis = 1)
        B = tf.concat([rs(Bx1), rs(Bx2), rs(Bx3), rs(Bx4)], axis = 1)
        
        return (A, B)




        
        
        
        
        
        

        