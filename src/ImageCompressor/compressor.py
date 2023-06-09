import numpy as np
import os
import scipy
import time
from . import image_functions as img_func


def compress_svd(img, r, algo):
    """Compress the image using 2-phase SVD with rank r"""
    # Use nbytes to get the size of the numpy array in bytes
    original_size = img.nbytes  

    # Initialize start time
    start_time = time.time()

    # If image is RGB   
    if len(img.shape) == 3:
        red_channel, green_channel, blue_channel =  img[:, :, 0], img[:, :, 1], img[:, :, 2]

        XR_r = img_func.truncate_svd(red_channel, 6, r, r, svd_version=algo)[0]
        XG_r = img_func.truncate_svd(green_channel, 6, r, r, svd_version=algo)[0]
        XB_r = img_func.truncate_svd(blue_channel, 6, r, r, svd_version=algo)[0]
        X_r = np.dstack((XR_r, XG_r, XB_r))

        compressed_image = X_r
    # If image is grayscale
    else:
        if algo == 'numpy':
            U, S, VT = img_func.numpy_svd(img)
        elif algo == 'phase_A':
            U, S, VT = img_func.phase_A_svd(img)
        elif algo == 'phase_B':
            U, S, VT = img_func.phase_B_svd(img)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    
        # If S's rank is less than k, adjust k
        # k = min(k, S.shape[0])
        
        # Truncate U, S, and VT using the rank-k approximation
        U_k = U[:, :r]
        S_k = S[:r, :r]
        VT_k = VT[:r, :]
    
        # Reconstruct the compressed image
        compressed_image = U_k @ S_k @ VT_k

    # Calculate compression time
    end_time = time.time()
    compression_time = end_time - start_time

    # Compute the size reduction of compressed image
    compressed_size = compressed_image.nbytes
    size_reduction = 100 * (original_size - compressed_size) / original_size
    
    return compressed_image, compression_time, size_reduction
