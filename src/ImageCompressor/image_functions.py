import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.sparse import diags
from matplotlib.pyplot import imread
from ..SVD.two_phase_svd import svd_phase_A, svd_phase_B


def img2double(img):
    """
    Converts image pixel intensities to double precision floating point numbers.

    Args:
    img (numpy.ndarray): Input image as an array.

    Returns:
    numpy.ndarray: Image with pixel intensities as double precision floating point numbers.
    """
    return img.astype(np.float64) 


def numpy_svd(img, full_matrices=True):
    """
    Performs SVD using numpy's built-in function.

    Args:
    img (numpy.ndarray): Input image as an array.
    full_matrices (bool): If True, compute full-sized U and V. If False (default), compute the compact forms.

    Returns:
    Tuple[numpy.ndarray]: U, Sigma and V^T matrices of the SVD.
    """
    U, S, VT = np.linalg.svd(img, full_matrices=full_matrices)
    return (U, np.diag(S), VT)


def phase_A_svd(img):
    """
    Performs SVD using the phase A algorithm.

    Args:
    img (numpy.ndarray): Input image as an array.

    Returns:
    Tuple[numpy.ndarray]: U, Sigma and V^T matrices of the SVD.
    """
    U, S, VT = svd_phase_A(img)
    return (U, np.diag(S), VT)


def phase_B_svd(img):
    """
    Performs SVD using the phase B algorithm.

    Args:
    img (numpy.ndarray): Input image as an array.

    Returns:
    Tuple[numpy.ndarray]: U, Sigma and V^T matrices of the SVD.
    """
    U, S, VT = svd_phase_B(img)
    return (U, np.diag(S), VT)


def frobenious_norm(A, B):
    """
    Calculates the Frobenius norm of the difference between two matrices.

    Args:
    A (numpy.ndarray): First input matrix.
    B (numpy.ndarray): Second input matrix.

    Returns:
    float: Frobenius norm of the difference between the two matrices.
    """
    return np.linalg.norm(A-B, ord='fro')


def get_optimal_rank_by_norm(X, max_norm):
    """
    Calculates the optimal rank of a matrix by minimizing the Frobenius norm.

    Args:
    X (numpy.ndarray): Input matrix.
    max_norm (float): Maximum allowable Frobenius norm.

    Returns:
    int: Optimal rank.
    """
    U, S, VT = numpy_svd(X)
    max_rank_ = S.shape[0]
    opt_rank_ = 1
    while True:
        X_r = U[:, :opt_rank_] @ S[:opt_rank_, :opt_rank_] @ VT[:opt_rank_, :]
        norm = frobenious_norm(X, X_r)
        if norm > max_norm:
            opt_rank_ += 1
            continue
        break 
    return opt_rank_


def get_optimal_rank(X, **kwargs):
    """
    Calculates the optimal rank of a matrix or a set of matrices.

    Args:
    X (numpy.ndarray or list[numpy.ndarray]): Input matrix or matrices.

    Returns:
    int: Optimal rank.
    """
    if len(X.shape) == 3:
        opt_rank_ = np.mean([
                get_optimal_rank_by_norm(X[:, :, 0], kwargs['max_norm']),
                get_optimal_rank_by_norm(X[:, :, 1], kwargs['max_norm']),
                get_optimal_rank_by_norm(X[:, :, 2], kwargs['max_norm']),
        ])    
    else:
        opt_rank_ = get_optimal_rank_by_norm(X, kwargs['max_norm'])
    return int(opt_rank_)


def tridiag(n, k):
    """
    Generates a tridiagonal matrix and returns its k-th power.

    Args:
    n (int): Size of the square matrix.
    k (int): Power to raise the matrix to.

    Returns:
    numpy.ndarray: k-th power of the tridiagonal matrix.
    """
    output = np.diag([1 / (4 + 0.1)] * (n - 1), -1) + np.diag([(2 + 0.1) / (4 + 0.1)] * n, 0) + np.diag([1 / (4 + 0.1)] * (n - 1), 1)
    return np.linalg.matrix_power(output, k).astype(np.float64)


def truncate(U, Sigma, VT, trunc):
    """
    Truncates the SVD of a matrix.

    Args:
    U (numpy.ndarray): U matrix of the SVD.
    Sigma (numpy.ndarray): Sigma matrix of the SVD.
    VT (numpy.ndarray): V^T matrix of the SVD.
    trunc (int): Rank at which to truncate.

    Returns:
    numpy.ndarray: Truncated matrix.
    """
    singular_values = np.diag(Sigma)
    A_trunc = np.zeros(U.shape)
    for i in range(trunc):
        u = U[:, i].reshape(-1,1)
        v = VT[i, :].reshape(-1,1)
        A_trunc += (v @ u.T) * (1 / singular_values[i])
    return A_trunc.astype(np.float64)


def PSNR(original, recovered):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between the original and the recovered matrix.

    Args:
    original (numpy.ndarray): Original matrix.
    recovered (numpy.ndarray): Recovered matrix.

    Returns:
    float: PSNR between the original and recovered matrix.
    """
    return 10 * np.log10((recovered.shape[0] ** 2) / np.linalg.norm(recovered-original, ord='fro'))


def truncate_svd(X, k, ltrunc, rtrunc, svd_version="numpy"):
    """
    Performs SVD truncation on the input matrix and calculates the PSNR and singular values.

    Args:
    X (numpy.ndarray): Input matrix.
    k (int): Power to raise the tridiagonal matrices to.
    ltrunc (int): Rank at which to truncate the left tridiagonal matrix.
    rtrunc (int): Rank at which to truncate the right tridiagonal matrix.
    svd_version (str): The version of SVD to use. Options are "numpy", "phase_A", "phase_B". Default is "numpy".

    Returns:
    list: List containing the truncated matrix, the PSNR, and the singular values.
    """
    Al = tridiag(X.shape[0], k)
    Ar = tridiag(X.shape[0], k)

    if svd_version == "numpy":
        U, Sigma, VT = numpy_svd(Al)
    elif svd_version == "phase_A":
        U, Sigma, VT = phase_A_svd(Al)
    elif svd_version == "phase_B":
        U, Sigma, VT = phase_B_svd(Al)
    else:
        raise ValueError("Invalid SVD version. Choose either 'numpy', 'phase_A', or 'phase_B'.")

    Al_trunc = truncate(U, Sigma, VT, ltrunc)
    Ar_trunc = truncate(U, Sigma, VT, rtrunc)
    B = Al @ X @ Ar 
    X_trunc = Al_trunc @ B @ Ar_trunc
    psnr_value = PSNR(X, X_trunc)
    return [X_trunc, psnr_value, np.diag(Sigma)]


def relative_error(true_matrix, approximate_matrix):
    """
    Calculates the relative error between the true and approximate matrices.

    Args:
    true_matrix (numpy.ndarray): True matrix.
    approximate_matrix (numpy.ndarray): Approximate matrix.

    Returns:
    float: Relative error between the two matrices.
    """
    absolute_error = np.abs(true_matrix - approximate_matrix)
    error_norm = np.linalg.norm(absolute_error, ord='fro')
    true_norm = np.linalg.norm(true_matrix, ord='fro')
    relative_error = error_norm / true_norm
    return relative_error


def load_image(filename, extension):
    """Load an image from the 'test_images' directory."""
    PATH = f"test_images/{filename}.{extension}"
    img = imread(PATH)
    plt.imshow(img)
    return img2double(img)


def to_grayscale(img):
    """Convert image to its grayscaled version"""
    img = img2double(img)
    img_gray = np.mean(img, -1)

    print(f'original image: {img.shape}')
    print(f'grayscaled image: {img_gray.shape}')

    # Plot orginal and grayscaled image
    fig = plt.figure(0, (12,6))
    for idx, im in enumerate([img, img_gray]):
        ax = plt.subplot(1,2, idx+1)
        if len(im.shape)==2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im)

    return img_gray # Output grayscale image


def process_and_plot_image(image_path):
    """Process and plot image as input"""
    if image_path[-3:] == "png":
        img = imread(image_path)
    else:
        img = Image.open(image_path)
        img = np.asarray(img)

    img = img2double(img)
    img_gray = np.mean(img, -1)

    n_rows, n_cols = img_gray.shape

    print(f'original image: {img.shape}')
    print(f'grayscaled image: {img_gray.shape}')

    fig = plt.figure(0, (12, 6))
    for idx, im in enumerate([img, img_gray]):
        ax = plt.subplot(1, 2, idx+1)
        if len(im.shape) == 2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im)

    return img, img_gray # Output original and grayscale image


def plot_grayscale_images(U: np.ndarray, S: np.ndarray, VT: np.ndarray, img_gray: np.ndarray, ranks: list = [5, 25, 50, 100, 250]):
    """Function to plot the images for different ranks and the original image"""
    fig = plt.figure(0, (18, 12))
    fig.subplots_adjust(top=1.1)

    for idx, r in enumerate(ranks):
        X_r = U[:, :r] @ S[:r, :r] @ VT[:r, :]

        ax = plt.subplot(2, 3, idx+1)
        ax.imshow(X_r, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'''rank {r}\nfrobenious_norm: {round(frobenious_norm(img_gray, X_r), 2)}''')
        
    ax = plt.subplot(2, 3, idx+2)
    ax.imshow(img_gray, cmap='gray')
    ax.set_title('original image')
    plt.show()
    
    
def plot_rank_v_sigma_and_frobenious_norm(U: np.ndarray, S: np.ndarray, VT: np.ndarray, img_gray: np.ndarray):
    """Function to plot the graphs for rank versus log sigma and rank versus frobenious norm"""
    fig = plt.figure(0, (12, 6))
    fig.subplots_adjust(top=1.7, right=1.)

    ax1 = plt.subplot(2, 2, 1)
    ax1.semilogy(np.diag(S))
    ax1.set_xlabel('rank')
    ax1.set_ylabel('log sigma')
    ax1.set_title('rank   v/s   log_sigma')

    frob_norm = []
    x_ticks = []
    rank = np.linalg.matrix_rank(img_gray)
    for r in np.linspace(1, rank, 100):
        r = int(r)
        x_ticks.append(r)

        X_r = U[:, :r] @ S[:r, :r] @ VT[:r, :]

        frob_norm.append(frobenious_norm(img_gray, X_r))

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x_ticks, frob_norm)
    ax2.set_xlabel('rank')
    ax2.set_ylabel('frobenious_norm')
    ax2.set_title('rank   v/s   frobenious_norm')
    plt.show()


def plot_truncate_svd(img_gray, S):
    """Function to plot the graphs for truncation versus log sigma for different truncation values"""
    figure, axis = plt.subplots(2, 3, figsize=(15, 15))

    # Use enumerate to get both the index and the value of trunc_values
    trunc_values = [5, 25, 50, 100, 250]
    for idx, trunc_value in enumerate(trunc_values):
        i = idx // 3
        j = idx % 3
        axis[i, j].semilogy(truncate_svd(img_gray, 6, trunc_value, trunc_value)[2])
        axis[i, j].set_title('trunc v/s log_sigma for trunc_value = {}'.format(trunc_value))

    axis[1, 2].semilogy(np.diag(S))
    axis[1, 2].set_title('trunc v/s log_sigma')
    plt.show()

    
def plot_psnr_values(img_gray: np.ndarray):
    """Function to plot the graphs for PSNR values versus truncation values """
    psnr_vals = []
    trunc_values = [5, 25, 50, 100, 250]

    for idx, r in enumerate(trunc_values):
        psnr_vals.append(truncate_svd(img_gray, 6, r, r)[1])

    plt.plot(trunc_values, psnr_vals,marker='o')
    plt.title("PSNR Values Plot")
    plt.xlabel("Trunc values")
    plt.ylabel("PSNR Values")
    plt.xticks(trunc_values, rotation ='vertical')
    plt.show()

    
def plot_image_channels(img: np.ndarray):
    """Function to plot the image channels for given truncation values"""
    trunc_vals = [5, 25, 50, 100, 250]
    red_channel, green_channel, blue_channel =  img[:, :, 0], img[:, :, 1], img[:, :, 2]

    fig = plt.figure(0, (18, 12))
    fig.subplots_adjust(top=1.1)

    for idx, r in enumerate(trunc_vals):
        XR_r = truncate_svd(red_channel, 6, r, r)[0]
        XG_r = truncate_svd(green_channel, 6, r, r)[0]
        XB_r = truncate_svd(blue_channel, 6, r, r)[0]
        X_r = np.dstack((XR_r, XG_r, XB_r))
        ax = plt.subplot(2, 3, idx+1)
        ax.imshow(X_r)
        ax.set_xticks([])
        ax.set_yticks([]) 
        ax.set_title(f'''Trunc: {r}\n''')

    ax = plt.subplot(2, 3, idx+2)
    ax.imshow(img)
    ax.set_title('original image')
    ax.set_xticks([])
    ax.set_yticks([])