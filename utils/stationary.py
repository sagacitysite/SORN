import warnings
import numpy as np

def calculate(transition):
    # Correct very small numbers to zero
    transition[np.abs(transition) < 1e-15] = 0
    
    # Calculate eigenvalues and eigenvectors
    eigvalvec = np.linalg.eig(transition.T)

    # Get position of eigenvalue, where eigenvalue equals 1
    eigval_pos = int(np.argwhere(np.isclose(eigvalvec[0], 1)))

    # Get eigenvectors for chosen eigenvalue and norm it, such that the sum equals 1
    vec = np.abs(eigvalvec[1][:, eigval_pos])
    return vec / np.sum(vec)

def transition_entropy(transition):
    P = transition
    # Correct very small numbers to zero
    P[np.abs(P) < 1e-15] = 0

    # Calculate stationary distribution
    m = calculate(P)

    H = 0
    for i in range(np.shape(P)[0]):
        for j in range(np.shape(P)[1]):
            # Only calculate entropy, if P_ij is non zero, otherwise numpy will throw an error
            if P[i,j] != 0:
                H += m[i]*P[i,j]*np.log(P[i,j])
    return -H
