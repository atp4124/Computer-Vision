import cv2

def denoise(image, kernel_type,ksize,sigma=0.5):
    if kernel_type == 'box':
        imOu=cv2.blur(image,ksize)
        
    elif kernel_type == 'median':
        imOu=cv2.medianBlur(image,ksize)
       
    elif kernel_type == 'gaussian':
        imOu=cv2.GaussianBlur(image,ksize,sigma)
    else:
        print('Operatio Not implemented')
    return imOu