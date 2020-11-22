import numpy as np
import cv2

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    h,w,_=input_image.shape
    new_image=np.zeros([h,w,4])
    # ligtness method

    new_image[:,:,0]=(np.max(input_image,axis=2)+np.min(input_image,axis=2))/2

    # average method

    new_image[:,:,1]= np.mean(input_image,axis=2)

    # luminosity method

    new_image[:,:,2]= np.dot(input_image[...,:3], [0.21, 0.72, 0.07])

    # built-in opencv function 

    new_image[:,:,3]=cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
    

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    R = input_image[:, :, 0]
    G = input_image[:, :, 1]
    B = input_image[:, :, 2]

    O1 = (R-G)/np.sqrt(2)
    O2 = (R+G-2*B)/np.sqrt(6)
    O3 = (R+G+B)/np.sqrt(3)

    new_image=cv2.merge((O1,O2,O3))

    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    R = input_image[:, :, 0]
    G = input_image[:, :, 1]
    B = input_image[:, :, 2]

    r=R/(R+G+B)*255
    g=G/(R+G+B)*255
    b=B/(R+G+B)*255

    new_image=cv2.merge((r,g,b))
     

    return new_image
