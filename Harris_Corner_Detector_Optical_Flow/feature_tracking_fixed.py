import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from harris_corner_detection import detect_corner, gray_float
import os


def feature_tracking(image_dir='./pingpong/',n=15,filter_sigma=3):
    files = os.listdir(image_dir)
    nfiles = len(files)
    kernel_x = np.array([[1., -1.]])
    kernel_y = kernel_x.T
    mode = 'same'
    files.sort()
    reference_frame = cv2.imread(os.path.join(image_dir, files[0]))
    H, y_coordinate, x_coordinate = detect_corner(reference_frame, sigma=2, threshhold=30, n=n)
    y_coordinate = y_coordinate.tolist()
    x_coordinate = x_coordinate.tolist()
    reference_frame = gray_float(reference_frame)
    plt.imshow(reference_frame, cmap='gray')
    plt.scatter( x_coordinate, y_coordinate, marker='o')
    plt.show()
    for i in range(1,nfiles):
        im = cv2.imread(os.path.join(image_dir, files[i]))
        im = gray_float(im)
        I_x = signal.convolve2d(im, kernel_x, mode=mode)
        I_y = signal.convolve2d(im, kernel_y, mode=mode)
        I_x=gaussian_filter(I_x,sigma=filter_sigma)
        I_y=gaussian_filter(I_y,sigma=filter_sigma)
        I_t = im - reference_frame
        I_t=gaussian_filter(I_t,sigma=filter_sigma)
        lst_coord = zip(y_coordinate, x_coordinate)
        new_pos_x = []
        new_pos_y = []
        for y,x in lst_coord:
                x_int = int(x)
                y_int = int(y)
                window_x=I_x[y_int-n:y_int+n+1,x_int-n:x_int+n+1].flatten()
                window_y=I_y[y_int-n:y_int+n+1,x_int-n:x_int+n+1].flatten()
                window_t=I_t[y_int-n:y_int+n+1,x_int-n:x_int+n+1].flatten()
                window_x= np.expand_dims(window_x, axis=1)
                window_y = np.expand_dims(window_y, axis=1)
                window_t = np.expand_dims(window_t, axis=1)
                A = np.concatenate((window_x, window_y), axis=1)
                b = -window_t
                vu = np.linalg.inv(A.T @ A) @ A.T @ b
                new_x = vu[0,0] + x
                new_y = vu[1,0] + y
                new_pos_x.append(new_x)
                new_pos_y.append(new_y)
        # Prepare for next iteration
        reference_frame = im
        x_coordinate = new_pos_x
        y_coordinate = new_pos_y
        plt.figure()
        plt.imshow(im,cmap='gray')
        plt.scatter(np.array(new_pos_x),np.array(new_pos_y),marker='o')
        plt.show()
        #your folder name where you want to save the images
        # plt.savefig('./person_toy_image/frame_{}.png'.format(i))
if __name__=='__main__':
    feature_tracking()