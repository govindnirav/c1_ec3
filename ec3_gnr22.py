#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from pyDOE import *
from ipywidgets import interactive
import ipywidgets as widgets
from memory_profiler import memory_usage
import timeit

#%%
# Question 3

# Define function
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Define parameters
a = 0.1
b = -0.13
c = 9
t = np.linspace(0, 1, 100)

# Calculate f(t) for these parameters
f_values = f(t, a, b, c)

# Def an interpolating function
g = interp1d(t, f_values, kind='linear')

# Finer time grid for evaluation
t_fine = np.linspace(0, 1, 1000)
f_interp = g(t_fine)
f_exact=f(t_fine, a, b, c)

# Plot the original samples and the interpolated function
plt.figure(figsize=(10, 6))
plt.plot(t_fine, f_interp, '-', label='Interpolated Curve', color='blue', linewidth=3)
plt.plot(t_fine, f_exact, '-', label='Exact Function', color='red', linewidth=1.5)
plt.xlabel('Time $t$')
plt.ylabel(r'$f(t)$')
plt.title('Original Samples and Interpolated Function')
plt.legend()
plt.grid(True)
plt.show()

#%% 
# Question 6
# Define function
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Define parameters
a = np.linspace(0,1,10)
b = -0.13
c = 9
t = np.linspace(0, 1, 100)

# Create an empty DataFrame to store the time series data
df = pd.DataFrame(index=t)

# Compute and store the time series for each a value
for i in a:
    f_values = f(t, i, b, c)
    df[f'a={i:.2f}'] = f_values 

g = interp1d(a, df.values, kind='linear', axis=1)

a_test = 0.125
f_exact = f(t, 0.125, b, c)
f_interp = g(a_test)

print(f_interp)
print(f_exact)

# Plotting the exact and interpolated curves
plt.figure(figsize=(12, 6))
plt.plot(t, f_interp, label=f'Interpolated Series for a={a_test:.3f}',color='blue', linewidth=1)
plt.plot(t, f_exact, label=f'Exact Series for a={a_test:.3f}', color='red', linewidth=1)
plt.xlabel('Time $t$')
plt.ylabel(r'$f(t)$')
plt.title(f'Interpolated Time Series for a={a_test:.3f}')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the ratios
ratio = f_interp / f_exact

plt.figure(figsize=(10, 6))
plt.plot(t, ratio, label='Ratio of Interpolated to Exact', color='purple')
plt.plot(t, f_exact, '-', label='Exact Function', color='blue', linewidth=1.5)
plt.xlabel('Time $t$')
plt.ylabel('Ratio/'r'$f(t)$')
plt.title(f'Ratio of Interpolated Values to True Values when a = {a_test}')
plt.legend()
plt.grid(True)
plt.show()


#%%
# Question 11
# Define function
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Defining parameters
c = 9
t = np.linspace(0, 1, 100)

# Irregularly spaced data points
# Obtaining 10 random values of a and b which are seeded and comply with range
np.random.seed(5)
N = 10
a_sample = lhs(1, samples=N).flatten()
b_sample = np.interp(lhs(1, samples=N).flatten(), (0, 1), (-0.5, 0.5))
#print(a_sample, b_sample)

# Create an empty DataFrame to store the time series data
f_sample = np.array([f(i, a_sample, b_sample, c) for i in t]) # Creates a time series of f values for every (a_sample, b_sample)

# Define a regular grid over which to interpolate
ai = np.linspace(0, 1, 50)
bi = np.linspace(-0.5, 0.5, 50)
a_grid, b_grid = np.meshgrid(ai, bi)

# Create an array to store the interpolated data
interp_df = pd.DataFrame(index=pd.MultiIndex.from_product([ai, bi], names=['a', 'b']))

# Interpolate the data for each time step
for i, j in enumerate(t):
    f_value = f_sample[i, :]  # Get values at this time step for all (a, b)
    f_interp = griddata(
        points = np.column_stack((a_sample, b_sample)), 
        values = f_value, 
        xi = (a_grid, b_grid), 
        method='cubic'
    )
    interp_df[f't={j:.2f}'] = f_interp.flatten()


#%% 
# Question 13

# Generate values using the original function
def exact_function():
    # Define a regular grid over which to interpolate
    ai = np.linspace(0, 1, 50)
    bi = np.linspace(-0.5, 0.5, 50)
    a_grid, b_grid = np.meshgrid(ai, bi)

    f_values = np.array([f(t, a, b, c) for a, b in zip(a_sample, b_sample)])
    return f_values

# Generate values using the interpolator
def interpolator(a_sample=a_sample, b_sample=b_sample, t=t):
    f_sample = np.array([f(i, a_sample, b_sample, c) for i in t]) # Creates a time series of f values for every (a_sample, b_sample)

    # Define a regular grid over which to interpolate
    ai = np.linspace(0, 1, 50)
    bi = np.linspace(-0.5, 0.5, 50)
    a_grid, b_grid = np.meshgrid(ai, bi)

    # Create an array to store the interpolated data
    interp_df = pd.DataFrame(index=pd.MultiIndex.from_product([ai, bi], names=['a', 'b']))

    # Interpolate the data for each time step
    for i, j in enumerate(t):
        f_value = f_sample[i, :]  # Get values at this time step for all (a, b)
        f_interp = griddata(
            points = np.column_stack((a_sample, b_sample)), 
            values = f_value, 
            xi = (a_grid, b_grid), 
            method='cubic'
        )
        interp_df[f't={j:.2f}'] = f_interp.flatten()
    
    return interp_df

# Measure time
time_original = timeit.timeit(original_function_call, number=1)
time_interpolator = timeit.timeit(interpolator_call, number=1)

# Measure memory
memory_original = max(memory_usage((original_function_call,)))
memory_interpolator = max(memory_usage((interpolator_call,)))

# Print results
print("Time Comparison:")
print(f"Original function call: {time_original:.6f} seconds")
print(f"Interpolator call: {time_interpolator:.6f} seconds")

print("\nMemory Usage Comparison:")
print(f"Original function call: {memory_original:.2f} MB")
print(f"Interpolator call: {memory_interpolator:.2f} MB")