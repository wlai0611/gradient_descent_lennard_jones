import numpy as np

def get_gradients(func, parameters, perturb=0.01):
    gradients = np.zeros(len(parameters))
    for param_number, parameter in enumerate(parameters):
        param_copy = parameters.copy()
        param_copy[param_number] += perturb
        forward_perturb          =  param_copy.copy()
        param_copy               =  parameters.copy()
        param_copy[param_number] -= perturb
        backward_perturb         =  param_copy.copy()
        gradients[param_number]  =  (func(forward_perturb) - func(backward_perturb))/(2*perturb)
    return gradients

def gradient_descent(func, starting_params, max_iter=100, step_len=0.1):
    fx      = func(starting_params)
    counter = 0
    params  = starting_params.copy()
    while fx > step_len and counter < max_iter:
        gradients            = get_gradients(func, params)
        normalized_gradients = gradients/np.sqrt(np.sum(gradients**2))
        step_direction = normalized_gradients
        step_vector    = step_len * step_direction
        params         = params - step_vector
        fx = func(params)
        counter += 1

    return params