import numpy as np
import cv2
from keypoint_matching import keypoint_matching
import matplotlib.pyplot as plt

def nearest_neighbors(i, j, M, T):
    '''
    :param i: x coordinate
    :param j: y coordinate
    :param M: source image
    :param T: transformation matrix
    '''
    x_max, y_max = M.shape[0] - 1, M.shape[1] - 1
    x, y, _ = T @ np.array([i, j, 1])


    if np.abs(x) > x_max:
        return [0, 0, 0]
    if np.abs(y) > y_max:
        return [0, 0, 0]

    if np.floor(x) == x and np.floor(y) == y and x>=0 and y>=0:
        x, y = int(x), int(y)
        return M[x, y]

    if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):
        x = int(np.floor(x))
    else:
        x = int(np.ceil(x))

    if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):
        y = int(np.floor(y))
    else:
        y = int(np.ceil(y))



    return M[x, y]


def affineTransformMatrix(kps1,kps2):
    """This function calculates a affine transform matrix, for an arbitrary
       number of given corresponding points. Shape of xs and ys should
       be [[x1, y1], [x2, y2], ..., [xn, yn]]."""

    # Assert the given arrays have the same size and contain at least four points.

    if len(kps1) != len(kps2) or len(kps1) < 3:
        print("Arrays should have same size and should containt at least 3 points")
        return

    # Make the matrix A and vector b.
    A = list()
    b = list()

    # Append two rows to A for every element of kps1/kps2.
    for i in range(len(kps1)):
        A.append([kps1[i][0], kps1[i][1], 0, 0, 1, 0])
        A.append([0, 0, kps1[i][0], kps1[i][1], 0, 1])
        b.append(kps2[i][0])
        b.append(kps2[i][1])

    A = np.array(A)
    b = np.array(b)

    # Calculate the affine transform using the least squares
    p = np.linalg.lstsq(A, b)[0]

    # Reshape p to make the matrix P representing the affine transform.
    [m1, m2, m3, m4, t1, t2] = p
    P = np.array([[m1, m2, t1], [m3, m4, t2], [0, 0, 1]])

    return P

def ransac(kps1, kps2, matches, num_points=3, threshhold=0.1, num_tests=100):
    '''

    :param kps1: keypoints of image 1
    :param kps2: keypoints of image 2
    :param matches: keypoint matches
    :param num_points: number of points to determine a transformation, 3 in our case
    :param threshhold: maximum error in euclidian norm to determine
                     inliers (for RANSAC).
    :param num_tests: number of iterations

    '''

    match_tuples = [(kps1[match.queryIdx].pt, kps2[match.trainIdx].pt) for match in matches]
    best_match = (np.identity(3), [])
    for _ in range(num_tests):
        print("Iteration {}".format(_))
        # Pick 4 points at random (can be altered)
        # Get their coordinates
        match_list_indices = np.random.choice(len(match_tuples), num_points)
        match_list = [match_tuples[index] for index in match_list_indices]
        kp_list1 = []
        kp_list2 = []
        for (pt1, pt2) in match_list:
            kp_list1.append(pt1)
            kp_list2.append(pt2)

        # Calculate affine transform
        P = affineTransformMatrix(kp_list1, kp_list2)

        # Determine inliers
        inliers = []
        for index, (pt1, pt2) in enumerate(match_tuples):
            res = P @ np.append(pt1, 1)
            if res[-1] == 0:
                # Should we have res[-1] == 0 because of a rounding error,
                # we set the vector to the below value.
                # This way it will never be counted as an inlier.
                res = np.array([-threshhold - 1, -threshhold - 1, -1])
            else:
                res = res / res[-1]
            diff = np.linalg.norm(res[:2]-pt2)
            if np.abs(diff) <= threshhold:
                inliers.append(index)

        # If this matrix has more inliers than our current best match replace it.
        if len(inliers) > len(best_match[1]):
            best_match= (P, inliers)

    # Calculate matrix based on all inliers
    (P, inliers) = best_match

    inliers1 = [match_tuples[index][0] for index in inliers]
    inliers2 = [match_tuples[index][1] for index in inliers]
    P = affineTransformMatrix(inliers1, inliers2)

    print(f'num inliers {len(inliers1)}')

    return P, [matches[index] for index in inliers]

def visualise_transforms(image1,image2,method,error_margin,frac_inliers,dx=0,dy=0):
    '''
    :param image1: source image
    :param image2: target image
    :param method: visualise through warpAffine('affine') or nearest neighbour interpolation('interpolation')
    :param error_margin: probability of not having three inliers when
                        picking random matches (determines accuracy
                        of perspective transform matrix).
    :param frac_inliers: estimated fraction of matches that are inliers.
    :param dx: translation in the x direction
    :param dy: translation in the y direction

    '''
    # Get the keypoints and calculate the matches
    # of the descriptors in the two images.
    matches, kps1, kps2 = keypoint_matching(image1, image2)
    # Determine the number of tests.
    min_tests = int(np.log(error_margin) / np.log(1 - frac_inliers ** 3))
    print("Number of iterations {}".format(min_tests))
    # Determine the affine matrix (RANSAC).
    P, correct_matches = ransac(kps1, kps2, matches, num_tests=min_tests)
    translate = np.array([[1, 0, dx],
                          [0, 1, dy],
                          [0, 0, 1]])

    # Translate after projective transform (multiply matrices)
    P = translate @ P
    P_inverse = np.linalg.inv(P)
    shape1 = estimate_bounding_box(image1,P[:2])
    shape2= estimate_bounding_box(image2,P_inverse)
    if method == 'affine':
        image_transformed1=cv2.warpAffine(image1,P[:2],dsize=shape2,flags=cv2.INTER_NEAREST)
        image_transformed2=cv2.warpAffine(image2,P[:2],dsize=shape1,flags=cv2.WARP_INVERSE_MAP|cv2.INTER_NEAREST)
        fig=plt.figure()
        plt.imshow(image_transformed1[:,:,::-1])
        plt.title('Transformation of Image 1')
        fig=plt.figure()
        plt.imshow(image_transformed2[:,:,::-1])
        plt.title('Transformation of Image 2')
        img3 = cv2.drawMatches(image1, kps1, image2, kps2, correct_matches,\
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure()
        plt.imshow(img3)
    if method == 'interpolation':
        T_inv = np.linalg.inv(P)
        img_nn = np.empty((shape2[0], shape2[1], 3), dtype=np.uint8)
        img_nn_2 = np.empty((shape1[0], shape1[1], 3), dtype=np.uint8)

        for i, row in enumerate(img_nn):
            for j, col in enumerate(row):
                img_nn[i, j, :] = nearest_neighbors(i, j, image1, P)

        for i, row in enumerate(img_nn_2):
            for j, col in enumerate(row):
                img_nn_2[i, j, :] = nearest_neighbors(i, j, image2, T_inv)
        plt.figure()
        plt.imshow(img_nn)
        plt.title('Transformation of image 1')
        plt.figure()
        plt.imshow(img_nn_2)
        plt.title('Transformation of image 2')

def estimate_bounding_box(image,transf_matrix):
    '''

    :param transf_matrix: transformation matrix
    :return:
    '''
    h,w=image.shape[:2]
    corners_bef = [np.array([[0],[0],[1]]),np.array([[w],[0],[1]]),np.array([[w],[h],[1]]),np.array([[0],[h],[1]])]
    for i in range(len(corners_bef)):
        new_point=transf_matrix@corners_bef[i]
        corners_bef[i]=new_point[:2]
    xmin = min(corners_bef,key=lambda x:x[0])[0][0]
    ymin = min(corners_bef,key=lambda x:x[1])[1][0]
    xmax = max(corners_bef,key=lambda x:x[0])[0][0]
    ymax = max(corners_bef,key=lambda x:x[1])[1][0]

    return (int(np.round(abs(xmax - xmin),0)), int(np.round(abs(ymax - ymin),0)))

if __name__=='__main__':
    image1 = cv2.imread('boat1.pgm')
    image2 = cv2.imread('boat2.pgm')
    visualise_transforms(image1,image2,'interpolation',0.01,0.5)
    visualise_transforms(image1, image2, 'affine', 0.01, 0.5)

