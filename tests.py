import numpy as np
import matplotlib.pyplot as plt
from functions import gradient_descent,get_gradients,coordinates_to_potential_sum,SimulationHelper


class LinReg():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def chi_sq(self, params):
        m, b = params
        return np.sum((y-m*x-b)**2)
        
n = 20
x = np.linspace(0,10, n)
rng= np.random.default_rng(seed=1)
true_slope = 12
true_intercept = 7
y = true_slope*x + true_intercept + rng.random(n)
regressor = LinReg(x, y)
starting_params = [2,1]
m,b = gradient_descent(regressor.chi_sq, starting_params, step_len=0.5)
print(f'Expected m:{true_slope} b:{true_intercept}, Observed m:{m} b:{b}')

coordinates = np.array([
[-1,2,0],[-3,20,0],[-12,14,0]
]
, dtype=float)

true_potential_sum = -2.8257776083489956e-06
#print("Expected POtential",coordinates_to_potential_sum(coordinates, func = lambda r:4*(r**-12-r**-6)))

simulator = SimulationHelper(coordinates.reshape(3,3))
print("OpenMM Potential",simulator.potential_function(coordinates.reshape(3,3)))


starting_params = rng.random(9)

def potential_wrapper(coordinates):
    return simulator.potential_function(coordinates.reshape((len(coordinates)//3,3)))
    

min_coordinates = gradient_descent(potential_wrapper, starting_params, step_len=0.01, max_iter=1000)
min_coordinates = min_coordinates.reshape(len(min_coordinates)//3,3)
starting_params = starting_params.reshape(len(starting_params)//3,3)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(*min_coordinates.T)
plt.show()