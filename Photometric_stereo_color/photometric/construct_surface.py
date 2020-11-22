# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 20:32:14 2020

@author: atask
"""
import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        for x in range(h):
            if x==0:
                height_map[x, 0] = q[x, 0]
            else:
                height_map[x,0] = height_map[x-1,0]+q[x,0]
            for y in range (h):
                for z in range (1,w):
                    height_map [y, z] = height_map[y, z-1] + p[y, z]
    elif path_type=='row':
        for y in range(w):
            if y==0:
                height_map[0,y] = p[0,y]
            else:
                height_map[0,y]= height_map[0,y-1]+p[0,y]
            for x in range(w):
                for z in range(1,h):
                    height_map[z,x] = height_map[z-1,w]+q[z,x]
    return height_map