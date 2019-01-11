import numpy as np
import matplotlib.pyplot as plt

def stationary_distribution(transition_matrix):
    # Correct very small numbers to zero
    transition_matrix[np.abs(transition_matrix) < 1e-15] = 0
    
    # Calculate eigenvalues and eigenvectors
    eigvalvec = np.linalg.eig(transition_matrix.T)

    # Get position of eigenvalue, where eigenvalue equals 1
    eigval_pos = int(np.argwhere(np.isclose(eigvalvec[0], 1)))

    # Get eigenvectors for chosen eigenvalue and norm it, such that the sum equals 1
    vec = np.abs(eigvalvec[1][:, eigval_pos])
    return vec / np.sum(vec)

def transition_entropy(transition):  # entropy rate
    P = transition
    # Correct very small numbers to zero
    P[np.abs(P) < 1e-15] = 0

    # Calculate stationary distribution
    m = stationary_distribution(P)

    H = 0
    for i in range(np.shape(P)[0]):
        for j in range(np.shape(P)[1]):
            # Only calculate entropy, if P_ij is non zero, otherwise numpy will throw an error
            if P[i,j] != 0:
                H += m[i]*P[i,j]*np.log(P[i,j])
    return -H

#ar = np.array([[0.25, 0.25, 0.25, 0.25],
#               [0.25, 0.25, 0.25, 0.25],
#               [0.25, 0.25, 0.25, 0.25],
#               [0.25, 0.25, 0.25, 0.25]])
#ar = np.array([[0.05, 0.85, 0.05, 0.05],
#               [0.05, 0.05, 0.85, 0.05],
#               [0.05, 0.05, 0.05, 0.85],
#               [0.85, 0.05, 0.05, 0.05]])
#st = transition_entropy(ar)
#print(st)


transitions = []
iterate = np.array([0., 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.23, 0.24, 0.245, 0.25])
for it in iterate:
    transitions.append([[it, 1.-(3*it), it, it],
                        [it, it, 1.-(3*it), it],
                        [it, it, it, 1.-(3*it)],
                        [1.-(3*it), it, it, it]])

it_signal = np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0]])
it_noise = np.array([[0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25]])

tt = np.array(transitions)

p2_signal = np.sqrt(np.sum(np.square(tt-it_signal), axis=(1,2)))
p2_noise = np.sqrt(np.sum(np.square(tt-it_noise), axis=(1,2)))

# Calculate entropy
transition_entropy = np.array([transition_entropy(t) for t in tt])

# Plot
plt.errorbar(transition_entropy, p2_signal, label='Signal error')
plt.errorbar(transition_entropy, p2_noise, label='Noise error')
plt.ylim(ymin=0)
plt.xlabel('entropy rate')
plt.ylabel('p2 error regarding signal/noise')
plt.savefig('correlation_errors_entropy_signalnoise.svg', format='svg', transparent=True)
plt.close()
