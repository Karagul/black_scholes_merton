"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""
from math import sqrt
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    return out

# The Wiener process parameter.
delta = 2
# Total time.
T = 10.0
# Number of steps.
N = 100
# Time step size
dt = T/N
# Number of realizations to generate.
m = 10
# Create an empty array to store the realizations.
x = numpy.empty((m,N+1))
# Initial values of x.
x[:, 0] = 50

brownian(x[:,0], N, dt, delta, out=x[:,1:])

t = np.linspace(0.0, N*dt, N+1)
for k in range(m):
    plot(t, x[k])
xlabel('time step', fontsize=16)
ylabel('index level', fontsize=16)
grid(True)
show()

