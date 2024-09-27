"""
CS 4391 Homework 2 Programming: Part 4 - non-local means filter
Implement the nlm_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j] using a non-local means filter
    # step 1: compute window_sizexwindow_size filter weights of the non-local means filter in terms of intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the pixel values in the search window
    # Please see slides 30 and 31 of lecture 6. Clarification: the patch_size refers to the size of small image patches (image content in yellow, 
    # red, and blue boxes in the slide 30); intensity_variance denotes sigma^2 in slide 30; the window_size is the size of the search window as illustrated in slide 31.
    # Tip: use zero-padding to address the black border issue. 

    # ********************************
    # Your code is here.
        # Get image dimensions
    rows, cols = img.shape

    # Padding the image to handle borders
    pad = patch_size // 2
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)

    # Padding for the search window
    window_pad = window_size // 2
    padded_img = np.pad(padded_img, window_pad, mode='constant', constant_values=0)

    # Precompute Gaussian weight for patches (spatial similarity)
    spatial_weights = np.zeros((patch_size, patch_size))
    for k in range(patch_size):
        for l in range(patch_size):
            x = k - pad
            y = l - pad
            spatial_weights[k, l] = math.exp(-(x**2 + y**2) / (2 * intensity_variance))

    # Non-Local Means filtering for each pixel
    for i in range(rows):
        for j in range(cols):
            # Extract the reference patch from the image
            ref_patch = padded_img[i + window_pad: i + window_pad + patch_size, 
                                   j + window_pad: j + window_pad + patch_size]
            
            # Variables to store the accumulated weight and pixel value
            filtered_pixel = 0
            weight_sum = 0

            # Search window
            for m in range(window_size):
                for n in range(window_size):
                    # Extract a patch from the search window
                    search_patch = padded_img[i + m: i + m + patch_size, 
                                              j + n: j + n + patch_size]

                    # Compute patch similarity (L2 norm squared)
                    patch_diff = ref_patch - search_patch
                    intensity_diff = np.sum((patch_diff**2) * spatial_weights)

                    # Compute the weight based on intensity difference
                    weight = math.exp(-intensity_diff / (2 * intensity_variance))
                    
                    # Accumulate the weighted pixel value
                    filtered_pixel += weight * padded_img[i + m + pad, j + n + pad]
                    weight_sum += weight

            # Normalize the filtered pixel by the total weight
            img_filtered[i, j] = filtered_pixel / (weight_sum + 1e-16)  # Avoid division by zero

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
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)