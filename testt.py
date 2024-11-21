#%%
import numpy as np
from scipy.interpolate import griddata
from pyDOE import lhs
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

#%%


# Interpolation time


np.random.seed(42)
n_samples = 10


# Define parameter space
a_uniform = np.linspace(0, 1, 50)  # Full parameter space for a
b_uniform = np.linspace(-0.5, 0.5, 50)  # Full parameter space for b
a_uniform_grid, b_uniform_grid = np.meshgrid(a_uniform, b_uniform)


# Latin Hypercube samples (irregular grid of a and b)
a_lhd = lhs(1, samples=n_samples).flatten()
b_lhd = (lhs(1, samples=n_samples).flatten() - 0.5)


# Time grid and constant
c = 9
t = np.linspace(0, 1, 100)  # Time grid


# Define the function f
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)




# original function
start = time.time()
time_f_values = np.array([f(a=a, b=b, t=t, c=c) for a, b in zip(a_lhd, b_lhd)])
end = time.time()
print("Original time:", end - start)


#Interpolated function
start = time.time()


# Evaluate the function at sampled points over all time steps
f_samples = np.array([f(ti, a_lhd, b_lhd, c) for ti in t])  # Shape: (len(t), n_samples)


# Interpolate over the full parameter space for each time step
f_interpolated_list = []
for t_idx in range(len(t)):
    # Interpolate over the full (a, b) space
    f_interpolated = griddata(
        points=np.array([a_lhd, b_lhd]).T,  # Irregular grid points
        values=f_samples[t_idx],  # Function values at t[t_idx]
        xi=(a_uniform_grid, b_uniform_grid),  # Full parameter space grid
        method="linear",  # Cubic interpolation
    )
    f_interpolated_list.append(f_interpolated)
end = time.time()
print("Interpolation time:", end - start)




import sys


# Measure memory usage for original function evaluations
original_size = sys.getsizeof(time_f_values)


# Measure memory usage for interpolated data
interpolated_size = sum(sys.getsizeof(f) for f in f_interpolated_list)


print("Memory usage for original function evaluations:", original_size, "bytes")
print("Memory usage for interpolated data:", interpolated_size, "bytes")




# Output
#Original time: 0.0021495819091796875
#Interpolation time: 0.4242863655090332
#Memory usage for original function evaluations: 8128 bytes
#Memory usage for interpolated data: 12800 bytes
#Unsurprisingly the original function time is faster, and requires less memory - would be expected as less computation is required. Memory would be higher for the interpolation method as requires creating a storing the griddata structure.

