import numpy as np
import time
import scipy


# PHASE I
def bidiag(A):
    """
    The function is used to convert a given matrix A to bidiagonal form B, using 
    the Golub-Kahan bidiagonalization procedure and Householder transformations.

    Args:
    A (numpy.ndarray): Input matrix

    Returns:
    numpy.ndarray: Bidiagonal form of A
    numpy.ndarray: Matrix Q, which represents the sequence of Householder transformations
    """
    
    row, col = np.shape(A)  # Shape of the input matrix
    q = np.identity(col)  # Identity matrix of same size as A
    x = np.copy(A)  # Copy of input matrix
    B = np.copy(A)  # Copy of input matrix, will be transformed to bidiagonal form

    for i in range(col):
        # LEFT HOUSEHOLDER TRANSFORMATION
        cur_row, cur_col = np.shape(x)
        a = x[:, 0]
        e = np.zeros(cur_row)
        e[0] = -np.linalg.norm(a) if a[0] <= 0 else np.linalg.norm(a)

        v  = np.transpose(a - e)
        I = np.identity(cur_row)
        H_v = I if np.linalg.norm(v) == 0 else I - (2/np.linalg.norm(v)**2 * np.outer(v, v))

        x = H_v @ x  # Update x

        # RIGHT HOUSEHOLDER TRANSFORMATION
        if cur_col != 1:
            b = x[0, 1:]
            e = np.zeros(cur_col - 1)
            e[0] = -np.linalg.norm(b) if b[0] <= 0 else np.linalg.norm(b)

            v = np.transpose(b - e)
            I_left = np.identity(cur_col - 1)
            H_v = I_left if np.linalg.norm(v) == 0 else I_left - (2/np.linalg.norm(v)**2 * np.outer(v, v))

            H_right = np.identity(cur_col)
            H_right[1:, 1:] = H_v

            h_row, h_col = H_right.shape
            x = x @ H_right

            curr_v = np.identity(col)
            curr_v[col - h_row:col, col - h_col:col] = H_right
            q = q @ curr_v

        B[i:row, i:col] = x
        x = B[i+1:row, i+1:col]

    return B, q


def get_eig_2by2(x):
    """
    This function computes the eigenvalues of a 2x2 matrix, and then returns the
    eigenvalue that is closest to the bottom-right element of the matrix. 
    This is typically used to compute the Wilkinson shift during the QR algorithm.

    Args:
    x (numpy.ndarray): Input 2x2 matrix

    Returns:
    float: The eigenvalue of the 2x2 matrix closest to the bottom-right element
    """
    m = (x[0][0] + x[1][1]) / 2
    p = x[0][0] * x[1][1] - x[0][1] * x[1][0]
    eig_1 = m + (m**2 - p)**(1/2)
    eig_2 = m - (m**2 - p)**(1/2)

    return eig_1 if abs(eig_1 - x[1][1]) < abs(eig_2 - x[1][1]) else eig_2


# PHASE II-A
def svd_phase_A(A):
    """
    This function implements the phase 2A of the SVD decomposition, which involves
    applying the QR iteration with Wilkinson shift and deflation on a given matrix A 
    to obtain its SVD.

    Args:
    A (numpy.ndarray): Input matrix

    Returns:
    numpy.ndarray: Matrix U of the SVD
    numpy.ndarray: Singular values Sigma of the SVD
    numpy.ndarray: Matrix V transpose of the SVD
    """

    # Check the shape of A
    tall_matrix = True
    row, col = np.shape(A)
    if row < col:
        tall_matrix = False
        A = np.transpose(A)
    row, col = np.shape(A)

    # Perform the Golub-Kahan bidiagonalization
    B, Q_bidiag = bidiag(A)
    B = B[:col, :col]  # Get the upper square submatrix of B

    # Compute B^T * B
    x = np.transpose(B) @ B
    eigenvectors = np.identity(col)
    eigenvalues = [0 for i in range(col)]
    iterations = 0

    for i in range(col-1):
        x = x[0:col-i, 0:col-i]
        while True:
            iterations += 1
            n = x.shape[0]
            sub_x = x[n-2:n, n-2:n]

            shift = get_eig_2by2(sub_x)

            # Check for convergence
            checker = x[col-1-i, 0:col-1-i]
            if np.linalg.norm(checker) <= 10e-12:
                break

            # Perform QR factorization with shift
            shift_matrix = shift * np.identity(n)
            q_matrix, r_matrix = np.linalg.qr(x - shift_matrix)
            x = r_matrix @ q_matrix + shift_matrix

            # Update eigenvectors
            q_fullsize = np.identity(col)
            q_fullsize[0:n, 0:n] = q_matrix
            eigenvectors = eigenvectors @ q_fullsize

        # Record eigenvalue
        eigenvalues[n-1] = x[col-1-i][col-1-i]

    # Record first eigenvalue
    eigenvalues[0] = x[0][0]

    # Compute V transpose
    V_t = np.transpose(eigenvectors) @ np.transpose(Q_bidiag)

    # Sort singular values and corresponding columns of V
    sigma = np.sqrt(np.array(eigenvalues))
    idx = sigma.argsort()[::-1]
    sigma = sigma[idx]
    V_t = np.transpose(V_t)
    V_t = V_t[:, idx]
    V_t = np.transpose(V_t)

    # Compute U using the formula Av = sigma * u
    u = np.zeros(shape=(row, row))
    for i in range(col):
        u[i, :] = A @ V_t[i] / sigma[i]

    # If A is short and wide, complete basis for U
    size_n = col
    for i in range(row):
        if size_n == row:
            break
        e = np.zeros(row)
        e[i] = 1
        v_temp = e
        for j in range(size_n):
            v_temp -= np.inner(e, u[j]) * u[j]

        if np.linalg.norm(v_temp) > 10e-12:
            u[size_n, :] = v_temp / np.linalg.norm(v_temp)
            size_n += 1

    U = np.transpose(u)

    # If the original A was short and wide, swap U and V
    if not tall_matrix:
        old_U = U
        old_V_t = V_t
        U = np.transpose(old_V_t)
        V_t = np.transpose(old_U)

    print("Total Iterations:", iterations)
    return U, sigma, V_t


# PHASE II-B
def svd_phase_B(A):
    """
    This function implements the phase 2B of the SVD decomposition, which involves
    applying an iterative procedure that coincides with the QR iteration for 
    the tridiagonal matrix B⊤B with zero shift on a given matrix A to obtain its SVD.

    Args:
    A (numpy.ndarray): Input matrix

    Returns:
    numpy.ndarray: Matrix U of the SVD
    numpy.ndarray: Singular values Sigma of the SVD
    numpy.ndarray: Matrix V transpose of the SVD
    """

    iterations = 0

    # Check the shape of A
    tall_matrix = True
    row, col = np.shape(A)
    if row < col:
        tall_matrix = False
        A = np.transpose(A)
    row, col = np.shape(A)

    # Perform the Golub-Kahan bidiagonalization
    B, Q_bidiag = bidiag(A)
    B = B[:col, :col]  # Get the upper square submatrix of B

    x = B
    eigenvectors = np.identity(col)
    eigenvalues = [0 for i in range(col)]

    # QR iteration with zero shift
    for i in range(col):
        x = x[0:col-i, 0:col-i]
        while True:
            iterations += 1
            n = x.shape[0]
            if n == 1:
                break

            # Perform QR factorization of X^T
            q, r = np.linalg.qr(np.transpose(x))
            
            # Update eigenvectors
            q_fullsize = np.identity(col)
            q_fullsize[0:n, 0:n] = q
            eigenvectors = eigenvectors @ q_fullsize

            # Perform Cholesky decomposition of R*R^T and update X
            l = np.linalg.cholesky(r @ np.transpose(r))
            x = np.transpose(l)
            if abs(x[0][1]) < 10e-12:
                break
        
        # Record eigenvalue
        eigenvalues[n - 1] = x[n-1][n-1]

    # Record last eigenvalue
    eigenvalues[0] = x[n - 2][n - 1]

    # Compute V transpose
    V_t = np.transpose(eigenvectors) @ np.transpose(Q_bidiag)

    # Compute U using the formula Av = sigma * u
    sigma = np.array(eigenvalues)
    u = np.zeros(shape=(row, row))
    for i in range(col):
        u[i, :] = A @ V_t[i] / eigenvalues[i]

    # If A is short and wide, complete basis for U
    size_n = col
    for i in range(row):
        if size_n == row:
            break
        e = np.zeros(row)
        e[i] = 1
        v_temp = e
        for j in range(size_n):
            v_temp -= (np.inner(e, u[j])) * u[j]

        if np.linalg.norm(v_temp) > 10e-12:
            u[size_n, :] = v_temp / np.linalg.norm(v_temp)
            size_n += 1

    U = np.transpose(u)

    # If the original A was short and wide, swap U and V
    if not tall_matrix:
        old_U = U
        old_V_t = V_t
        U = np.transpose(old_V_t)
        V_t = np.transpose(old_U)

    print("Total Iterations:", iterations)
    return U, sigma, V_t


# Time comparison between Phase II-A and II-B
# if __name__ == "__main__":
#     Test_Matrix = np.random.rand(300, 300)
#     start = time.time()
#     s,v,d = svd_phase_A(Test_Matrix)
#     end = time.time()
#     print("Time for Phase IIA = ", end - start)

#     start = time.time()
#     s,v,d = svd_phase_B(Test_Matrix)
#     end = time.time()
#     print("Time for Phase IIB = ", end - start)