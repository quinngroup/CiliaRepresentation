import numpy as np

# generate (20000,10) numpy array of random numbers over normal distribution
sigma = 5
mu = 0
starting_points = sigma * np.random.randn(20000,10) + mu
np.save("starting_points", starting_points)

# create functions over R10 vector field
def vector_field(v):
    return np.array([.5*v[0] + 4 * math.sin(v[7]),
                     1.2*math.exp(v[1]) - 2 / (np.linalg.norm(v) + 1),
                     .6 * math.pow(v[3], 1.5) + v[1] * v[9],
                     v[3]/(math.pow(v[2], 2) + .01) + .9 * v[8],
                     v[5]*v[3]*v[6]/math.pow(math.cos(v[7]), 2),
                     -1 * np.linalg.norm(v),
                     .2 * (v[9] - v[0]) + 2 + v[0]*math.pow(v[4],v[1]),
                     -3.6*math.cos(v[8]) + math.sqrt(math.pow(v[1], 2) + math.pow(v[3] - v[4], 2)),
                     -.4*math.sin(math.pow(v[2], 2)) + .6 * math.exp(v[5]),
                     v[3]/(np.linalg.norm(v) + math.pow(v[3] - v[7], 2) + 1)])

# use scipi neural integrator to generate paths 

