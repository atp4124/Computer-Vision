import cv2
import matplotlib.pyplot as plt

def image_formation(albedo_image,shading_image):
    original_image=cv2.multiply(albedo_image/255,shading_image/255)
    
    return original_image

if __name__ == '__main__':
    # Replace the image name with a valid image
    image_path_albedo = 'ball_albedo.png'
    image_path_shading = 'ball_shading.png'
    
    # Read with opencv
    albedo = cv2.imread(image_path_albedo)
    shading = cv2.imread(image_path_shading)
    ball=cv2.imread('ball.png')
    # Convert from BGR to RGB
    # This is a shorthand.
    albedo = albedo[:, :, ::-1]
    shading = shading[:,:,::-1]
    ball = ball[:,:,::-1]
    original_image = image_formation(albedo,shading)
    plt.figure()
    plt.imshow(ball)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(albedo)
    ax2.imshow(shading)
    ax3.imshow(original_image)
