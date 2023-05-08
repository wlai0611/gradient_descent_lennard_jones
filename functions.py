import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *

def get_gradients(func, parameters, perturb=0.0001):
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

def coordinates_to_potential_sum(coordinates, func = lambda r:4*(r**-12-r**-6)):
    potential_sum = 0
    coordinates = coordinates.reshape((len(coordinates)//2,2))

    for atom_i in range(len(coordinates)):
        per_dimension_distances = coordinates[atom_i,:] - coordinates
        interatomic_distances   = np.sqrt(np.sum(per_dimension_distances**2,axis=1))
        interatomic_distances[np.isclose(interatomic_distances,0)] = 1
        atom_i_potentials = func(interatomic_distances)
        potential_sum += atom_i_potentials.sum()

    return potential_sum/2 #its symmetric, we double counted

class SimulationHelper():
    def __init__(self, initial_coordinates):
        self.natoms, self.ndims = initial_coordinates.shape
        self.set_atoms(element="argon", atomic_number=18, mass=1*amu)
        self.set_forces(sigma= 1*nanometer, epsilon=1*kilocalories_per_mole)
        self.simulation = Simulation(     self.topology, self.system,
                                      LangevinIntegrator(293.15*kelvin,1/picosecond,2*femtoseconds))
        #self.simulation.context.setPositions(initial_coordinates*nanometer)

    def set_atoms(self, element, atomic_number, mass):
        self.system = System()
        self.topology = Topology()
        chain  = self.topology.addChain()
        residue= self.topology.addResidue(element,chain)
        element_object = Element.getByAtomicNumber(atomic_number)

        for atom_index in range(self.natoms):
            self.system.addParticle(mass)
            self.topology.addAtom(element, element_object,residue)
    
    def set_forces(self, sigma, epsilon, charge=0.0):
        self.force = openmm.NonbondedForce()
        for atom_index in range(self.natoms):
            self.force.addParticle(charge, sigma, epsilon)
        self.system.addForce(self.force)

    def potential_function(self, coordinates):
        self.simulation.context.setPositions(coordinates*nanometer)
        state = self.simulation.context.getState(getEnergy=True)
        energy= state.getPotentialEnergy()/kilocalories_per_mole
        return energy
    
    def potential_wrapper(self, coordinates):
        coordinates = coordinates.reshape((len(coordinates)//2,2))
        energy = self.potential_function(coordinates)
        return energy
