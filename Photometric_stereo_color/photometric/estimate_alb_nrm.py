# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:25:44 2020

@author: atask
"""
import numpy as np

def estimate_alb_nrm(image_stack, scriptV, shadow_trick=True):
    
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])

  
    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """
    
    for v in range(h):
        for j in range(w):
            i_dim=image_stack[v,j,:]
            i=np.expand_dims(i_dim,axis=1)
            if shadow_trick:    
                scripti=np.diag(i_dim)
                b=np.linalg.lstsq(scripti@np.float32(scriptV),scripti@i)[0]
                albedo[v,j]=np.linalg.norm(b)
                normal[v,j]= b.flatten()/(np.linalg.norm(b)+1e-7)
            else:
                b=np.linalg.lstsq(scriptV,i)[0]
                albedo[v,j]=np.linalg.norm(b)
                normal[v,j] = b.flatten()/(np.linalg.norm(b)+1e-7)
    return albedo, normal

