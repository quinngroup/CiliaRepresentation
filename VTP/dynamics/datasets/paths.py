import numpy as np
import math
from scipy.integrate import odeint

DATASET_SIZE = 50
APP_LSD = 10
DYN_LSD = 3
NUM_POINTS = 23

# generate (20000,10) and (20000,3) numpy array of random numbers over normal distribution

sigma = 5
mu = 0
starting_points = sigma * np.random.randn(DATASET_SIZE, APP_LSD) + mu
np.save("starting_points", starting_points)
latent_reps = np.random.randn(DATASET_SIZE, DYN_LSD)
np.save("latent_reps", latent_reps)

# create functions over R10 vector field
def vector_field(v, w):
    return np.array([.5 * v[0] + 4 * math.sin(v[7]),
                     1.2*math.exp(v[1]) - 2 / (np.linalg.norm(v) + 1) + w[2],
                     .6 * math.pow(v[3], 3) + v[1] * v[9],
                     v[3]/(math.pow(v[2] + w[0], 2) + .01) + .9 * v[8],
                     v[5]*v[3]*v[6]/math.pow(math.cos(v[7] * w[2]), 2),
                     -1 * np.linalg.norm(v),
                     .2 * (v[9] - v[0] * w[1]) + 2 + v[0]*math.pow(v[4] - v[1], 4),
                     -3.6*math.cos(v[8]) + math.sqrt(math.pow(v[1], 2) + math.pow(v[3] - v[4], 2)),
                     -.4*math.sin(w[1] + math.pow(v[2], 2)) + .6 * math.exp(v[5]),
                     v[3]/(np.linalg.norm(v) + math.pow(v[3] - v[7] + w[0], 2) + 1)])

# use scipy integrator to generate paths 
starting_points = np.load("starting_points.npy")
latent_reps = np.load("latent_reps.npy")
times = np.linspace(0, 1, NUM_POINTS)
def integrate(v, w):
    func = lambda y, t: vector_field(y, w)
    return odeint(func, v, times, full_output = 0)

paths = np.zeros((DATASET_SIZE, NUM_POINTS, APP_LSD))
for i in range(DATASET_SIZE):
    paths[i] = integrate(starting_points[i], latent_reps[i])
print(paths)

