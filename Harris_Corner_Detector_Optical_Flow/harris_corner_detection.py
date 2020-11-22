import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
# Are you allowed to use rotate
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cv2

def gray_float(image):
    if len(image.shape) == 3:
        im_gf = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
    else:
        im_gf = image.astype(float)
    
    return im_gf

def detect_corner(image, sigma=2, n=10, threshhold=30):
    im_gf = gray_float(image)

    I_x = gaussian_filter(im_gf, sigma, order=(0,1))
    I_y = gaussian_filter(im_gf, sigma, order=(1,0))

    plt.imshow(I_x, cmap='gray')
    plt.show()
    plt.imshow(I_y, cmap='gray')
    plt.show()

    I_xx = I_x**2
    I_yy = I_y**2
    I_xy = I_x*I_y

    # Smooth
    A = gaussian_filter(I_xx, sigma)
    C = gaussian_filter(I_yy, sigma)
    B = gaussian_filter(I_xy, sigma)

    # Harris response
    k = 0.04
    H = (A*C - B**2) - k*(A + C)**2

    # Find highest values
    h, w = H.shape
    r = []
    c = []
    for y in range(n,h - n):
        for x in range(n,w - n):
            window = H[y-n:y+n+1,x-n:x+n+1]
            value = window[n,n]
            assert value == H[y,x]
            if value > threshhold and value == np.max(window):
                print(value)
                r.append(y)
                c.append(x)
    return H, np.array(r), np.array(c)

def three_plots(image, r, c, sigma=2):
    # Plot derivatives
    plt.imshow(image)
    plt.show()
    im_gf = gray_float(image)

    I_x = gaussian_filter(im_gf, sigma, order=(0,1))
    I_y = gaussian_filter(im_gf, sigma, order=(1,0))
    plt.figure()
    plt.imshow(I_x, cmap='gray')
    plt.title('x derivative')
    plt.show()
    plt.figure()
    plt.title('y derivative')
    plt.imshow(I_y, cmap='gray')
    plt.show()

    # Plot image with corners
    plt.figure()
    plt.title('Image with corners')
    plt.imshow(image)
    plt.scatter(c, r, marker='o')
    plt.show()


if __name__ == "__main__":
    file_path = './person_toy/00000001.jpg'
    # file_path = './pingpong/0000.jpeg'
    im = cv2.imread(file_path)
    # im = rotate(im, 45)
 
    H, r, c = detect_corner(im)
    assert len(r) == len(c)
    three_plots(im, r, c)