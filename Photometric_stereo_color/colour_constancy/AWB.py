
from PIL import Image
import numpy as np

ori_image = Image.open('/home/isualice/UvA/lab1/colour_constancy/awb.jpg')


def AWB(ori_image):

    image = np.asarray(ori_image)

    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]

    # Take the average value of each color
    avgR = np.mean(np.mean(R))
    avgG = np.mean(np.mean(G))
    avgB = np.mean(np.mean(B))

    avgRGB = [avgR, avgG, avgB]

    # The total average over all three colors which will be close to gray
    # according to the assumption
    grayValue = (avgR + avgG + avgB)/3

    # Scale tje gray color value with respect to the average values
    scaleValue = grayValue/avgRGB


    newI = np.zeros((ori_image.size[1],ori_image.size[0],3), dtype=np.uint8)

    # To prevent the value over 255, we take the lower value of the two
    newI[:,:,0] = np.minimum(scaleValue[0] * R,255);
    newI[:,:,1] = np.minimum(scaleValue[1] * G,255);
    newI[:,:,2] = np.minimum(scaleValue[2] * B,255);
    new_img = Image.fromarray(newI, 'RGB')
    new_img.save('corrected.jpg')
    new_img.show()











