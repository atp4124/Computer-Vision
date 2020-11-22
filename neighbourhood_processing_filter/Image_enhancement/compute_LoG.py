from scipy.signal import convolve2d
import numpy as np
import cv2


def create_log(sigma,x,y):
    a=y**2+x**2-2*sigma**2
    b=2*np.pi*sigma**6
    c=np.exp(-(x**2+y**2)/(2*sigma**2))
    return a*c/b

def compute_LoG(image, LOG_type):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if LOG_type == 1:
        #method 1
        image=cv2.GaussianBlur(image/255,ksize=(5,5),sigmaY=0.5, sigmaX=0.5)
        laplacian_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        imOut=convolve2d(image,laplacian_kernel,mode='same')
        
    elif LOG_type == 2:
        #method 2
        size=5
        log_mask=np.zeros((size,size))
        w_range=int(np.floor(size/2))
        for i in range(-w_range,w_range+1):
            for j in range(-w_range,w_range+1):
                log_mask[i,j]=create_log(sigma=0.5,x=i,y=j)
        log_mask=log_mask/(np.sum(log_mask))
        print(np.sum(log_mask))
        imOut=convolve2d(image,log_mask,mode='same')
  

    elif LOG_type == 3:
        #method 3
        sigma1=0.5
        sigma2=0.2
        image1=cv2.GaussianBlur(image,ksize=(5,5), sigmaY=sigma1,sigmaX=sigma1)
        image2=cv2.GaussianBlur(image,ksize=(5,5), sigmaY=sigma2,sigmaX=sigma2)
        imOut=image1/(np.sum(image1))-image2/(np.sum(image2))

    return imOut

