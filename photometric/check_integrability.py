# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:23:22 2020

@author: atask
"""

import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])
    
    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy
    
    """
    
    #diffs = np.diff(normals)
    p = normals[:, :, 0] / (normals[:, :, 2]+np.ones(normals.shape[:2])*1e-7)
    q = normals[:, :, 1] / (normals[:, :, 2]+np.ones(normals.shape[:2])*1e-7)
    
    
    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0
    
    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    
    """
    
    p2, _ = np.gradient(p)
    _, q2 = np.gradient(q)
    
    
    SE = (p2 - q2)**2
    
    
    
    return p, q, SE

 
