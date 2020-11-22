import cv2
import numpy
import matplotlib.pyplot as plt
import random


def keypoint_matching(im1, im2):
    # Create sift
    sift = cv2.SIFT_create()
    kps1, dscs1 = sift.detectAndCompute(im1, mask=None)
    kps2, dscs2 = sift.detectAndCompute(im2, mask=None)
    matcher = cv2.BFMatcher()
    matches = matcher.match(dscs1, dscs2)

    return matches, kps1, kps2


if __name__=='__main__':
    image1=cv2.imread('boat1.pgm')
    image2=cv2.imread('boat2.pgm')
    image1=image1[:,:,::-1]
    image2=image2[:,:,::-1]
    matches, kps1, kps2=keypoint_matching(image1,image2)
    matches=random.sample(matches,10)
    img3=cv2.drawMatches(image1, kps1, image2, kps2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(img3)
    plt.show()
