import cv2
import numpy as np
import matplotlib.pyplot as plt
from keypoint_matching import keypoint_matching
from ransac import ransac, estimate_bounding_box

def stitch_images(im1, im2,shape_method, frac_inliers=0.5, error_margin=0.01,dx=0,dy=0,shape_predifined=(600,700)):
    """
    Stitch two images im1 and im2 together.
    Parameters:
        - frac_inliers: estimated fraction of matches that are inliers.
        - error_margin: probability of not having three inliers when
                        picking random matches (determines accuracy
                        of perspective transform matrix).
        - threshold: maximum error in euclidian norm to determine
                     inliers (for RANSAC).
        - dx: translation in x direction of images after applying
              affine transform.
        - dy: translation in y direction of images after applying
              affine transform.
        - shape_method: 'function' for using an algorithm to determine the size of stiched image
                        'precalculated' for specifing a size for stitched image
        - shape_predefined

    """
    # Get the keypoints and calculate the matches
    # of the descriptors in the two images.
    matches, kps1, kps2 = keypoint_matching(im1, im2)

    # Determine the number of tests.
    min_tests = int(np.log(error_margin) / np.log(1 - frac_inliers ** 3))

    # Determine the projective matrix (RANSAC and SVD tric).
    P, _ = ransac(kps1, kps2, matches, num_tests=min_tests)
    if P is None:
        print("Not enough inliers found")
        return
    # Set up translation matrix
    translate = np.array([[1, 0, dx],
                          [0, 1, dy],
                          [0, 0, 1]])

    # Translate after projective transform (multiply matrices)
    P = translate @ P
    P_inverse = np.linalg.inv(P)
    if shape_method=='function':
             shape = estimate_bounding_box(im2, P_inverse)
    elif shape_method=='precalculated':
             shape=shape_predifined
    # Warp second image.
    f_stitched = cv2.warpAffine(im1, P[:2], dsize=shape)
    M, N = im2.shape[:2]
    print(f'M,N : {M, N}')
    # Paste im1 onto the stiched image in the upper left corner
    # and apply same translation as image 1.
    dx = np.abs(dx);
    dy = np.abs(dy)
    f_stitched[dy:M + dy, dx:N + dx, :] = im2

    # Show image.
    fig = plt.gcf()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    plt.imshow(f_stitched[:, :, ::-1])
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    im1_left=cv2.imread('left.jpg')
    im2_right=cv2.imread('right.jpg')
    stitch_images(im1_left, im2_right,shape_method='precalculated',frac_inliers=0.2,dx=300,dy=100,error_margin=0.02)
    stitch_images(im1_left, im2_right, shape_method='function', frac_inliers=0.2)
    stitch_images(im1,im2,dx=50,dy=50)