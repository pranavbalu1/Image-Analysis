"""
CS 4391 Homework 2 Programming: Part 1&2 - mean and gaussian filters
Implement the linear_local_filtering() and gauss_kernel_generator() functions in this python script
"""

import cv2
import numpy as np
import math
import sys
 
def linear_local_filtering(
    img: np.uint8,
    filter_weights: np.ndarray,
) -> np.uint8:
    """
    Homework 2 Part 1
    Compute the filtered image given an input image and a kernel 
    """

    img = img / 255
    img = img.astype("float32") # input image
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    kernel_size = filter_weights.shape[0] # filter kernel size
    sizeX, sizeY = img.shape

    # filtering for each pixel
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):

            # Todo: For current position [i, j], you need to compute the filtered pixel value: img_filtered[i, j] 
            # using the kernel weights: filter_weights and the neighboring pixels of img[i, j] in the kernel_sizexkernel_size local window
            # The filtering formula can be found in slide 3 of lecture 6

            # ********************************
            # Your code is here.
            #Local Window at [i,j]
            local_window = img[i - kernel_size // 2: i + kernel_size // 2 + 1,
                                j - kernel_size // 2: j + kernel_size // 2 + 1]
    
            # Filter x Local Window
            filtered_pixel = np.sum(local_window * filter_weights)
            
            # Assign the filtered pixel to the output image
            img_filtered[i, j] = filtered_pixel

            # ********************************


    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    """
    Homework 2 Part 2
    Create a kernel_sizexkernel_size gaussian kernel of given the variance. 
    """
    # Todo: given variance: spatial_variance and kernel size, you need to create a kernel_sizexkernel_size gaussian kernel
    # Please check out the formula in slide 15 of lecture 6 to learn how to compute the gaussian kernel weight: g[k, l] at each position [k, l].
    kernel_weights = np.zeros((kernel_size, kernel_size))

    #Center of the kernel
    center = kernel_size // 2

    #constants varaince and coefficient
    variance = spatial_variance
    coefficient = 1 / (2 * math.pi * variance)

    #go through each pixel in the kernel
    for m in range(kernel_size):
        for n in range(kernel_size):
            k = m - center
            l = n - center
        
            #gaussian function
            exponent = - (k**2 + l**2) / (2 * variance)
            kernel_weights[m, n] = coefficient * math.exp(exponent)
    
    # Normalize the kernel -> sum of all weights now equals 1
    kernel_weights = kernel_weights / np.sum(kernel_weights)
    return kernel_weights
 
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
    
    # mean filtering
    box_filter = np.ones((7, 7))/49
    img_avg = linear_local_filtering(img_noise, box_filter) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_box.png', img_avg)

    # Gaussian filtering
    kernel_size = 7  
    spatial_var = 15 # sigma_s^2 
    gaussian_filter = gauss_kernel_generator(kernel_size, spatial_var)
    gaussian_filter_normlized = gaussian_filter / (np.sum(gaussian_filter)+1e-16) # normalization term
    im_g = linear_local_filtering(img_noise, gaussian_filter_normlized) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_gaussian.png', im_g)