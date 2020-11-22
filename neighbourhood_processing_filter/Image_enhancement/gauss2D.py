import numpy as np
from gauss1D import gauss1D


def gauss2D(sigma, kernel_size):
    
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    Gx = gauss1D(sigma, kernel_size)
    Gy = gauss1D(sigma, kernel_size)
    
    G2 = Gx.reshape(kernel_size, 1) @ Gy.reshape(1, kernel_size)
    
    return G2

if __name__ == '__main__':
    G2 = gauss2D(2, 5)
    print(np.sum(G2))
