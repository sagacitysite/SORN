import warnings
import numpy as np

def calculate(transition):
    # Initalize
    initial = np.array([0.4,0.4,0.1,0.1])
    step = initial
    num_steps = 100000

    # Simulate num_steps
    for i in range(num_steps):
        step = np.dot(step, transition)

    if np.array_equal(step, initial):
        warnings.warn("For current transition matrix, stationary distributions seems not to exist. Choose another one or change initial values.")

    return step

def entropy_rate(transition):
    P = transition
    m = calculate(transition)

    H = 0
    for i in range(np.shape(P)[0]):
        for j in range(np.shape(P)[1]):
            # FIXME Fix dirty hack!! How to handle P[i,j] = 0 ??
            if P[i,j] == 0:
                P[i,j] = 0.000001
            H += m[i]*P[i,j]*np.log(P[i,j])
    return -H
