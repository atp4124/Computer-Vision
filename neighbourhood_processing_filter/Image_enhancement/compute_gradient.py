from scipy.signal import convolve2d
import numpy as np
import cv2

def compute_gradient(image):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Kx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Ky=Kx.T
    Gx=convolve2d(image,Kx,mode='same')
    Gy=convolve2d(image,Ky,mode='same')
    im_magnitude=np.sqrt(Gx**2+Gy**2)
    im_magnitude=im_magnitude/np.sum(im_magnitude)
    im_direction=np.arctan(Gx/Gy)
    return Gx, Gy, im_magnitude,im_direction

