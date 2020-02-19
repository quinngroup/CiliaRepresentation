import numpy as np

# generate (1000,10) numpy array of random numbers over normal distribution
sigma = 1
mu = 0
starting_points = sigma * np.random.randn(1000,10) + 1
np.save("starting_points", starting_points)

# create functions over R10 vector field
functions = {} 

# use scikit neural integrator to generate paths 

