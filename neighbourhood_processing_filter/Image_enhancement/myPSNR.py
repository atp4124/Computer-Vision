import numpy as np
import cv2
import matplotlib.pyplot as plt 

def myPSNR( orig_image, approx_image ):
    orig_image=orig_image/255
    approx_image=approx_image/255
    mse=np.mean((orig_image-approx_image)**2)
    maximum_pixel=np.max(orig_image)
    PSNR=20*np.log10(maximum_pixel/np.sqrt(mse))
    return PSNR

