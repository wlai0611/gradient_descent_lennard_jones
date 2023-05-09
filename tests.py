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
print('Expected Potential', true_potential_sum)

simulator = SimulationHelper(coordinates.reshape(3,3))
print("OpenMM Potential",simulator.potential_function(coordinates.reshape(3,3)))

starting_params = rng.random(6)
steps, fxs = gradient_descent(coordinates_to_potential_sum, starting_params, step_len=0.01, max_iter=100, tolerance=0.01)
steps = np.array(steps)
nsteps, nparams = steps.shape
steps = steps.reshape(nsteps, nparams//2, 2)
fig, ax = plt.subplots(nrows=2)
ax[0].set(xlim=(0,2),ylim=(-1,2), title='Gradient Descent of the Potential Energy Based on Coordinates')
atoms = ax[0].scatter(steps[0,:,0], steps[0,:,1])
ax[1].set(xlim=(0,len(fxs)), ylim=(0,max(fxs)))
energies, = ax[1].plot([],[])
ax[1].set(xlabel='Timestep',ylabel='Energy')

def animate(i):
    current_step = steps[i]
    atoms.set_offsets(current_step[:,:2])
    energies.set_data(np.arange(i),fxs[:i])
    return atoms, energies

anim = anm.FuncAnimation(fig, animate, frames = len(steps), blit=True)
plt.show()
print()

starting_params = np.array([
    [0,0.01,0],
    [-0.1,0,0],
    [0.1,0,0]
]).flatten()


def potential_wrapper(coordinates):
    return simulator.potential_function(coordinates.reshape((len(coordinates)//3,3)))
    

steps, fxs = gradient_descent(potential_wrapper, starting_params, step_len=0.01, max_iter=200, tolerance=0.01)
steps = np.array(steps)
nsteps, nparams = steps.shape
steps = steps.reshape(nsteps, nparams//3, 3)

#https://stackoverflow.com/questions/41602588/how-to-create-3d-scatter-animations
fig    = plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=1, subplot_kw=dict(projection="3d"))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
axes[0].view_init(elev=-119, azim=-146, roll=-20)
axes[0].set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
atoms, = axes[0].plot(*steps[0].T,linestyle='',marker='o')
energies, = axes[1].plot(range(len(steps)), fxs, 0)
axes[1].ticklabel_format(axis='y',style='plain')
axes[1].view_init(elev=101,azim=-92, roll=0)

def animate(frame_number):
    coordinates = steps[frame_number]
    atoms.set_data(coordinates[:,0], coordinates[:,1])
    atoms.set_3d_properties(coordinates[:,2])
    return atoms,

anim = anm.FuncAnimation(fig, animate, len(steps), blit=True)
plt.show()

print()

