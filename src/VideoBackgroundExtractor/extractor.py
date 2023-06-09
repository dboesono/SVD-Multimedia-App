import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time


def input_video(path,video_cap=10):
    """
    Input video frames into a matrix

    Args:
        path (str): Path of the video file.
        video_cap (int, optional): Frame capture rate. Default is 10.

    Returns:
        tuple: Video matrix, video height, video width
    """
    cap = cv2.VideoCapture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    frame_number = int(cap.get(7))
    video_mat = np.zeros((frame_number//video_cap,height,width,3), dtype='float16')
    count = 0
    while (cap.isOpened()):
        a,b = cap.read()
        if a == False:
            cap.release()
        else:
            if count % video_cap == 0:
                b = b.astype("float16")/255
                video_mat[count//video_cap-1] = b
            count += 1
    return video_mat, height, width


def rgb2gray(video):
    """
    Convert a color video into grayscale.

    Args:
        video (np.array): Color video matrix.

    Returns:
        np.array: Grayscale video matrix.
    """
    frame_number,height,width,color = np.shape(video)
    gray_mat = np.zeros((frame_number,height,width), dtype='float16')
    for i in range(frame_number):
        gray_mat[i] = 0.114*video[i,:,:,0] + 0.587*video[i,:,:,1] + 0.299*video[i,:,:,2]
    return gray_mat
    

def rgb(video):
    """
    Separate the color channels of a video.

    Args:
        video (np.array): Color video matrix.

    Returns:
        tuple: Separate matrices for each color channel.
    """
    frame_number,height,width,color = np.shape(video)
    mat_0 = np.zeros((frame_number,height,width),dtype = 'float16')
    mat_1 = np.zeros((frame_number,height,width),dtype = 'float16')
    mat_2 = np.zeros((frame_number,height,width),dtype = 'float16')
    for i in range(frame_number):
        mat_0[i] = video[i,:,:,0]
    for i in range(frame_number):
        mat_1[i] = video[i,:,:,1]
    for i in range(frame_number):
        mat_2[i] = video[i,:,:,2]
    return mat_0, mat_1, mat_2


def resize(mat):
    """
    Reshape each frame into a one-dimensional vector.

    Args:
        mat (np.array): Video matrix.

    Returns:
        np.array: Resized matrix.
    """
    frame_number,height,width= np.shape(mat)
    A_mat = np.zeros((frame_number,height*width),dtype = 'float16')
    for i in range(frame_number):
        resized = mat[i]
        A_mat[i] = resized.ravel()
    A_mat = np.transpose(A_mat)
    return A_mat


def power_iteration(mat):
    """
    Execute power iteration to find the dominant eigenvector.

    Args:
        mat (np.array): Input matrix.

    Returns:
        tuple: Dominant eigenvalue and corresponding eigenvector.
    """
    iter = 0
    eigenvalue_0 = 0
    height, width = np.shape(mat)
    x  = np.transpose(np.zeros((1,width)))
    x[0] = 1
    def power_iteration_execute(mat,x,eigenvalue_0): 
        x_0 = x
        x  = np.dot(mat,x)
        
        eigenvalue_1 =  np.linalg.norm(x)/np.linalg.norm(x_0)
        x  = x/np.linalg.norm(x)
        if abs(eigenvalue_1-eigenvalue_0)<= 10**-10:
            return eigenvalue_1, x
        else:
            return(power_iteration_execute(mat,x,eigenvalue_1))
    eigenvalue,x_output= power_iteration_execute(mat,x,eigenvalue_0)
    return eigenvalue, x_output


def SVD_largest(mat):
    """
    Calculate the largest singular value and corresponding singular vectors.

    Args:
        mat (np.array): Input matrix.

    Returns:
        tuple: Singular vectors and largest singular value.
    """
    v_eigen, v_vector = power_iteration(np.dot(np.transpose(mat),mat))
    u_vector = np.dot(mat, v_vector)/(v_eigen**0.5)
    return u_vector,v_eigen,v_vector


def get_background(mat,height,width):
    """
    Construct the background image.

    Args:
        mat (np.array): Input matrix.
        height (int): Original video height.
        width (int): Original video width.

    Returns:
        np.array: The constructed background image.
    """
    mat = mat.astype(float)
    U, sig, V = SVD_largest(mat)
    B = (sig**0.5)* U * V[0]
    B = B.reshape(height,width)
    return B
    
    
def monochrome(path):
    """
    Construct the background image for a grayscale video.

    Args:
        path (str): Path of the video file.
    """
    video, height, width= input_video(path)
    gray_video = rgb2gray(video)
    A_mat = resize(gray_video)
    print(np.shape(A_mat))
    B = get_background(A_mat,height,width)
    plt.figure(1)
    plt.axis('off')
    plt.gray()
    plt.imshow(B)
    plt.show()
    plt.close()


def mixed_color(path):
    """
    Construct the background image for a color video.

    Args:
        path (str): Path of the video file.

    Returns:
        B (np.array): The extracted background information matrix
    """
    video,height,width = input_video(path)
    A0,A1,A2 = rgb(video)
    A_mat_0 = resize(A0)
    A_mat_1 = resize(A1)
    A_mat_2 = resize(A2)
    B_0= get_background(A_mat_0,height,width)
    B_1= get_background(A_mat_1,height,width)
    B_2= get_background(A_mat_2,height,width)

    B = np.zeros((height,width,3))
    zeros = np.zeros((np.shape(B_0)))
    B[:,:,0],B[:,:,1],B[:,:,2] = B_2,B_1,B_0  

    # Plot the extracted background 
    # plt.figure(1)
    # plt.axis('off')
    # plt.imshow(B)
    # plt.show()

    return B # Output the extracted background