import numpy as np
import matplotlib.pyplot as plt
from functions import gradient_descent,get_gradients,coordinates_to_potential_sum,SimulationHelper
import matplotlib.animation as anm

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
steps, fxs = gradient_descent(regressor.chi_sq, starting_params, step_len=0.5)
m, b  = steps[-1]
print(f'Expected m:{true_slope} b:{true_intercept}, Observed m:{m} b:{b}')

coordinates = np.array([
[-1,2,0],[-3,20,0],[-12,14,0]
]
, dtype=float)

true_potential_sum = -2.8257776083489956e-06
#print("Expected POtential",coordinates_to_potential_sum(coordinates, func = lambda r:4*(r**-12-r**-6)))

simulator = SimulationHelper(coordinates.reshape(3,3))
print("OpenMM Potential",simulator.potential_function(coordinates.reshape(3,3)))

'''
starting_params = np.array([
    [0,0.01,0],
    [-0.1,0,0],
    [0.1,0,0]
]).flatten()
'''


def potential_wrapper(coordinates):
    return simulator.potential_function(coordinates.reshape((len(coordinates)//3,3)))
    

steps, fxs = gradient_descent(potential_wrapper, starting_params, step_len=0.01, max_iter=100, tolerance=3)
steps = np.array(steps)
nsteps, nparams = steps.shape
steps = steps.reshape(nsteps, nparams//3, 3)

#https://stackoverflow.com/questions/41602588/how-to-create-3d-scatter-animations
fig    = plt.figure()
ax     = plt.axes(projection='3d')
ax.view_init(elev=166, azim=-144, roll=-20)
ax.set(xlim=(-2,2), ylim=(-2,2), zlim=(-2,2))
atoms, = ax.plot(*steps[0].T,linestyle='',marker='o')

def animate(frame_number):
    coordinates = steps[frame_number]
    atoms.set_data(coordinates[:,0], coordinates[:,1])
    atoms.set_3d_properties(coordinates[:,2])
    return atoms,

anim = anm.FuncAnimation(fig, animate, len(steps), blit=True)
plt.show()

print()