# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 02:18:18 2020

@author: nitin
"""
import numpy as np
import scipy.io as sp
import tensorflow.compat.v1 as tf

class PhasePortrait(object):
    
    @staticmethod
    def phasePortrait(fname, xrange, yrange, f):
        X, Y = np.meshgrid(xrange, yrange)
        Z = f(X, Y);
        data = dict(X = X, Y = Y, Z = Z);
        sp.savemat(fname, data)
    
    @staticmethod
    def thetaVariedFormatter(theta, tdot, middleFunction):
        tsz = theta.shape[1]
        tdotsz = tdot.shape[0]
        totalsz = tsz * tdotsz
        allpoints = np.concatenate((
            np.zeros((totalsz, 1)), 
            np.zeros((totalsz, 1)), 
            np.reshape(theta, (totalsz, 1)), 
            np.reshape(tdot, (totalsz, 1))), 
            axis = 1);
        v = middleFunction(allpoints);
        output = np.reshape(v, (tdotsz, tsz))
        return output
    
    @staticmethod
    def thetaVariedPhasePortrait(fname, cartpole, valueFunction):
        tsz = 100; tdotsz = 200; cp = cartpole;
        PhasePortrait.phasePortrait(
            fname,
            np.linspace(-1*np.pi, np.pi, tsz),
            np.linspace(- cp.g * cp.L , cp.g * cp.L, tdotsz),
            lambda t, tdot : PhasePortrait.thetaVariedFormatter(t, tdot, valueFunction))
    
    @staticmethod
    def uncontrolledThetaDDPhasePortrait(fname, cartpole):
        def v(data):
            A, B = cartpole.f(data)
            return A[:, 3]
        PhasePortrait.thetaVariedPhasePortrait(fname, cartpole, v);

    @staticmethod
    def valueFunctionPhasePortrait(fname, cartpole, fbsnn):
        def v(data):
            samples = data.shape[0];
            time = tf.zeros((samples, 1))
            tf_dict = {fbsnn.Xi_tf: data, fbsnn.t_tf: time, fbsnn.W_tf: np.zeros(samples, 1)}
            
            valueph, gradvalueph = fbsnn.net_u(time, dataTensor);
            
            values = self.sess.run(valueph, tf_dict)
            gradvalues = self.sess.run(gradvalueph, tf_dict)
            
            return values;
        PhasePortrait.thetaVariedPhasePortrait(fname, cartpole, v)
            
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