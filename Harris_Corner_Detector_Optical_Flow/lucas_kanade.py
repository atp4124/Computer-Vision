from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def optical_flow(image1, image2, filter_sigma=0.7,window_size=15):
    '''
    Optical flow algorithm implemented for Part 2
    :param image1: reference image
    :param image2: second frame
    :param filter_sigma: standard deviation for Gaussian Filter
    :param window_size: window size for optical flow
    :return: optical flow vectors for each region
    '''

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(float)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY).astype(float)
    image1 = image1
    image2 = image2
    kernel_x=np.array([[1.,-1.]])
    kernel_y=kernel_x.T
    mode = 'same'
    I_x=signal.convolve2d(image2,kernel_x,mode=mode)
    I_y=signal.convolve2d(image2,kernel_y,mode=mode)
    I_x = gaussian_filter(I_x, sigma=filter_sigma)
    I_y = gaussian_filter(I_y, sigma=filter_sigma)
    I_t = image2 - image1
    I_t = gaussian_filter(I_t, sigma=filter_sigma)
    k, m= image1.shape
    no_regions=k//window_size
    u = []
    v = []
    blocks_I_x=np.array([I_x[i:i+window_size, j:j+window_size] for j in range(0,k-window_size,window_size) \
                     for i in range(0,k-window_size,window_size)])
    blocks_I_y=np.array([I_y[i:i+window_size, j:j+window_size] for j in range(0,k-window_size,window_size) \
                     for i in range(0,k-window_size,window_size)])
    blocks_I_t=np.array([I_t[i:i+window_size, j:j+window_size] for j in range(0,k-window_size,window_size) \
                     for i in range(0,k-window_size,window_size)])
    for i in range(blocks_I_x.shape[0]):
        fx=blocks_I_x[i,:,:].flatten()
        fy=blocks_I_y[i,:,:].flatten()
        ft=blocks_I_t[i,:,:].flatten()
        fx = np.expand_dims(fx, axis=1)
        fy = np.expand_dims(fy, axis=1)
        ft = np.expand_dims(ft, axis=1)
        A = np.concatenate((fx, fy), axis=1)
        b = -ft
        Vu=np.linalg.inv(A.T@A)@A.T@b
        u.append(Vu[0,0])
        v.append(Vu[1,0])


    u=np.array(u).reshape((no_regions,no_regions))
    v =np.array(v).reshape((no_regions, no_regions))
    return u,v

def optical_flow_tracking(image1,image2,sigma_value,old_positions,n):
    '''
    Optical flow function for Feature Tracking, Part 3
    :param image1: reference frame
    :param image2: next frames
    :param sigma_value: standard deviation of Gaussian filter
    :param old_positions: old positions of the pixels
    :param n: window size
    :return: optical flow vectors
    '''
    kernel_x = np.array([[1., -1.]])
    kernel_y = kernel_x.T
    mode = 'same'
    I_x = signal.convolve2d(image2, kernel_x, mode=mode)
    I_y = signal.convolve2d(image2, kernel_y, mode=mode)
    I_x = gaussian_filter(I_x, sigma=sigma_value)
    I_y = gaussian_filter(I_y, sigma=sigma_value)
    I_t = image2 - image1
    I_t = gaussian_filter(I_t, sigma=sigma_value)
    new_pos_x = []
    new_pos_y = []
    for y,x in old_positions:
        x_int = int(x)
        y_int = int(y)
        window_x = I_x[y_int - n:y_int + n + 1, x_int - n:x_int + n + 1].flatten()
        window_y = I_y[y_int - n:y_int + n + 1, x_int - n:x_int + n + 1].flatten()
        window_t = I_t[y_int - n:y_int + n + 1, x_int - n:x_int + n + 1].flatten()
        window_x = np.expand_dims(window_x, axis=1)
        window_y = np.expand_dims(window_y, axis=1)
        window_t = np.expand_dims(window_t, axis=1)
        A = np.concatenate((window_x, window_y), axis=1)
        b = -window_t
        vu = np.linalg.inv(A.T @ A) @ A.T @ b
        new_x = vu[0, 0] + x
        new_y = vu[1, 0] + y
        new_pos_x.append(new_x)
        new_pos_y.append(new_y)
    return np.array(new_pos_x), np.array(new_pos_y)

if __name__=='__main__':
   sphere1 = cv2.imread('sphere1.ppm')
   sphere1 = sphere1[:, :, ::-1]
   sphere2 = cv2.imread('sphere2.ppm')
   sphere2 = sphere2[:, :, ::-1]
   u, v = optical_flow(sphere1, sphere2, 0.5)
   x = np.arange(0,185,15)
   y = np.arange(0,185,15)
   X,Y = np.meshgrid(x,y)
   plt.figure()
   plt.quiver(X,Y,u, v,angles='xy',scale_units='xy',scale=0.5)
   plt.show()
   synth1 = cv2.imread('synth1.pgm')
   synth1 = synth1[:, :, ::-1]
   synth2 = cv2.imread('synth2.pgm')
   synth2 = synth2[:, :, ::-1]
   u_2, v_2 = optical_flow(synth1, synth2, 0.5)
   x = np.arange(0, 113, 15)
   y = np.arange(0, 113, 15)
   X, Y = np.meshgrid(x, y)
   plt.figure()
   plt.quiver(X, Y, u_2, v_2, angles='xy', scale_units='inches', scale=3)
   plt.show()


