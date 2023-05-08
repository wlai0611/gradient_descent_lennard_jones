import numpy as np
import matplotlib.pyplot as plt
from functions import gradient_descent, get_gradients

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

