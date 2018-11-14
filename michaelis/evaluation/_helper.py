from evaluation import *

def stationary_distribution(transition):
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


def calc_gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def norm_stationaries(estimated_stationaries):
    print('# Norm stationaries')
    
    # Normalize and return
    return estimated_stationaries / np.sum(estimated_stationaries, axis=4)[:, :, :, :, np.newaxis]

def calc_stationary_distances(est_norm):
    print('# Calculate stationary distances')

    # Calculate stationaries
    T = PARA.c.source.transitions
    stationaries = np.stack([stationary_distribution(trans) for trans in T])

    # Calculate distances
    # runs, models, train steps, thresholds, test chunks, stationary
    diff = est_norm - stationaries[np.newaxis, :, np.newaxis, np.newaxis, :]
    distances = np.sum(np.square(diff), axis=4)

    return distances

def prepare_hamming(hamming_distances):
    print('# Prepare hamming')
    # runs, models, train steps, hammings

    # Separate in test chunks
    chunks = PARA.c.steps_noplastic_test/PARA.c.stats.transition_step_size
    hamming_distances = np.moveaxis(np.split(hamming_distances, chunks, axis= 3), 0, 3)
    # Form after: runs, models, train steps, test chunks, hammings

    # Calculate hamming means
    return np.mean(hamming_distances, axis=4) / PARA.c.N_e

