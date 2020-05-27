import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from sella import Sella
import math
import itertools
from .atoms_png_render import render_image


MOVE_ACTION_NAMES = [
    'up',
    'down',
    'left',
    'right',
    'forward',
    'backward']

MOVE_ACTION = np.stack([
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0])])

ACTION_LOOKUP = [
    'move',
    'minimize',
#     'transition_state_search',
    'steepest_descent',
    'steepest_ascent']

ELEMENT_LATTICE_CONSTANTS = {'Ag': 4.124, 'Au': 4.153, 'Cu': 3.626}


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, size=(2, 2, 4),
                 element_choices={'Ag': 6, 'Au': 5, 'Cu': 5},
                 permute_seed=42,
                step_size=0.1):

        self.step_size=step_size
        
        self.initial_atoms = self._generate_slab(
            size, element_choices, permute_seed)

        # Mark the free atoms
        self.free_atoms = list(set(range(len(self.initial_atoms))) -
                               set(self.initial_atoms.constraints[0].get_indices()))

        # Set up the initial atoms
        self.reset()

        # Define the possible actions
        self.action_space = spaces.Tuple((spaces.Discrete(len(ACTION_LOOKUP)),
                                          spaces.Discrete(
                                              len(self.free_atoms)),
                                          spaces.Discrete(len(MOVE_ACTION))))

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=self._get_state().shape)  

        return

    # open AI gym API requirements
    def step(self, action):

        action_type = ACTION_LOOKUP[action[0]]

        if action_type == 'move':
            atom_index = action[1]
            move_index = action[2]
            self._move_atom(atom_index, move_index)

        elif action_type == 'minimize':
            self._minimize()

        elif action_type == 'transition_state_search':
            self._transition_state_search()

        elif action_type == 'steepest_descent':
            atom_index = action[1]
            self._steepest_descent(atom_index)

        elif action_type == 'steepest_ascent':
            atom_index = action[1]
            self._steepest_ascent(atom_index)

        else:
            raise Exception('I am not sure what action you mean!')

        self.atoms.wrap()
        
        observation = self._get_state()
        
        relative_energy = self.atoms.get_potential_energy() - self.initial_energy

        reward = self._get_reward(relative_energy)
        
        if relative_energy > 2:
            episode_over = True
        else:
            episode_over = False
    
        return observation, reward, episode_over, {}

    def reset(self):
        self.atoms = self.initial_atoms.copy()
        self.atoms.set_calculator(EMT())
        self.initial_energy = self.atoms.get_potential_energy()
        self.highest_energy = 0.0
        return self._get_state()

    def render(self, mode='rgb_array'):
        
        if mode=='rgb_array':
            # return an rgb array representing the picture of the atoms
            return render_image(self.atoms.repeat((2,2,1)), rotation='48x,-51y,-144z', bbox=(-10,0,10,20))
#         return
    
    
    def close(self):
        return

    # Helper functions for above
    def _minimize(self):
        dyn = BFGSLineSearch(atoms=self.atoms, logfile=None)
        dyn.run(0.1)
        return

    def _get_reward(self, relative_energy):        
        reward = 0
        
        if relative_energy < 0:
            # we found a better minima! great!
            reward += -relative_energy
        
        if relative_energy > self.highest_energy:
            # we just went over a higher transition state! bad!
            reward += -(relative_energy - self.highest_energy)
            self.highest_energy = relative_energy
            
        return reward

    def _transition_state_search(self):
        fix = self.atoms.constraints[0].get_indices()

        dyn = Sella(self.atoms,  # Your Atoms object
                    constraints=dict(fix=fix),  # Your constraints
                    trajectory='saddle.traj',  # Optional trajectory
                    )
        dyn.run(1e-2, steps=10)

        return

    def _steepest_descent(self, atom_index):
        force = self.atoms.get_forces()[self.free_atoms[atom_index], :]
        move = -self.step_size*force/np.linalg.norm(force)
        self.atoms.positions[self.free_atoms[atom_index]] += move
        return

    def _steepest_ascent(self, atom_index):
        force = self.atoms.get_forces()[self.free_atoms[atom_index], :]
        move = self.step_size*force/np.linalg.norm(force)
        self.atoms.positions[self.free_atoms[atom_index]] += move
        return

    def _move_atom(self, atom_index, move_index):
        # Helper function to move an atom
        self.atoms.positions[self.free_atoms[atom_index]
                             ] += MOVE_ACTION[move_index]*self.step_size
        return

    def _get_state(self):
        # helper function to get the current state space, which is just the position
        # of the free atoms as one long vector (should be improved)
        return self.atoms.get_scaled_positions()[self.free_atoms]

    def _generate_slab(self, size, element_choices, permute_seed):
        # generate a pseudo-random sequence of elements
        np.random.seed(permute_seed)
        num_atoms = np.prod(size) # math.prod is only available in python 3.8
        atom_ordering = list(itertools.chain.from_iterable(
            [[key]*element_choices[key] for key in element_choices]))
        element_list = np.random.permutation(atom_ordering)

        # Use vergard's law to estimate the lattice constant
        a = np.sum([ELEMENT_LATTICE_CONSTANTS[key] *
                    element_choices[key]/num_atoms for key in element_choices])

        # Generate a base FCC slab
        slab = fcc111('Al', size=size, a=a, periodic=True, vacuum=10.0)
        slab.set_chemical_symbols(element_list)

        # Set the calculator
        slab.set_calculator(EMT())

        # Constrain the bottom two layers
        c = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < np.mean(
            slab.positions[:, 2])])  # Fix two layers
        slab.set_constraint(c)

        # Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=slab, logfile=None)
        dyn.run(0.1)

        return slab