"""
CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j]
    # step 1: compute kernel_sizexkernel_size spatial and intensity range weights of the bilateral filter in terms of spatial_variance and intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the neighboring pixels of img[i, j] in the kernel_sizexkernel_size local window
    # The bilateral filtering formula can be found in slide 15 of lecture 6
    # Tip: use zero-padding to address the black border issue.

    # ********************************
    # Your code is here.
    # Size of the original image
    rows, cols = img.shape

    # Padding the image to handle borders
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)

    # Precompute the spatial Gaussian kernel
    g_kernel = np.zeros((kernel_size, kernel_size))
    for k in range(kernel_size):
        for l in range(kernel_size):
            x = k - pad
            y = l - pad
            g_kernel[k, l] = math.exp(-(x**2 + y**2) / (2 * spatial_variance))
    
    # Bilateral filtering for each pixel
    for i in range(rows):
        for j in range(cols):
            local_window = padded_img[i:i + kernel_size, j:j + kernel_size]

            intensity_diff = local_window - img[i, j]

            intensity_weight = np.exp(-(intensity_diff**2) / (2 * intensity_variance))

            bilateral_weights = g_kernel * intensity_weight

            img_filtered[i, j] = np.sum(bilateral_weights * local_window) / np.sum(bilateral_weights)

    # ********************************
    
    
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)