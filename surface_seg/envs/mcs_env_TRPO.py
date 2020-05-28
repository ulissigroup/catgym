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
from surface_seg.envs.mcs_env import MCSEnv


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


class MCSEnv_TRPO(MCSEnv):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, size=(2, 2, 4),
                 element_choices={'Ag': 6, 'Au': 5, 'Cu': 5},
                 permute_seed=42,
                 step_size=0.1):
        
        super(MCSEnv_TRPO, self).__init__()

        self.step_size=step_size
        
        self.initial_atoms = self._generate_slab(
            size, element_choices, permute_seed)

        # Mark the free atoms
        self.free_atoms = list(set(range(len(self.initial_atoms))) -
                               set(self.initial_atoms.constraints[0].get_indices()))

        # Set up the initial atoms
        self.reset()
       
        self.action_space = spaces.MultiDiscrete([len(ACTION_LOOKUP), len(self.free_atoms), len(MOVE_ACTION)])

        self.observation_space = self._get_state_space()

        return

    # open AI gym API requirements
    def step(self, action):

        action_type = ACTION_LOOKUP[action[0]]

        if action_type == 'move':
            self._move_atom(action[1], 
                            action[2])

        elif action_type == 'minimize':
            self._minimize()

        elif action_type == 'transition_state_search':
            self._transition_state_search()

        elif action_type == 'steepest_descent':
            self._steepest_descent(action[1])

        elif action_type == 'steepest_ascent':
            self._steepest_ascent(action[1])

        else:
            raise Exception('I am not sure what action you mean!')

        relative_energy = self._get_relative_energy()

        observation = self._get_state()

        reward = self._get_reward(relative_energy)
        
        #Stop if relative energy gets too high
        if relative_energy > 2.0:
            episode_over = True
        else:
            episode_over = False
    
        return observation, reward, episode_over, {}

    def _move_atom(self, atom_index, move_index):
        # Helper function to move an atom
        self.atoms.positions[self.free_atoms[atom_index]] \
                        += MOVE_ACTION[move_index]*self.step_size
        return
    
    def _get_state_dict(self):
        # helper function to get the current state space, which is just the position
        # of the free atoms as one long vector (should be improved)
        return {'positions': self.atoms.get_scaled_positions()[self.free_atoms],
                'energy':np.array([self._get_relative_energy()]),
               'forces':self.atoms.get_forces()[self.free_atoms, :]}
    
    def _get_state(self):
        observation = self._get_state_dict()
        observation_energy = observation['energy']
        observation_positions = observation['positions'].flatten()
        observation_forces = observation['forces'].flatten()
        
        return np.hstack([observation_energy, observation_positions, observation_forces])
        
    def _get_state_space(self):
        observation = self._get_state_dict()
        observation_energy = observation['energy']
        observation_positions = observation['positions'].flatten()
        observation_forces = observation['forces'].flatten()
        
        E_low, E_high = -2, 2      
        pos_low, pos_high = 0, 1
        F_low, F_high = -2, 2
        
        state_shape = np.hstack([observation_energy, observation_positions, observation_forces]).shape
        
        low = np.hstack([np.repeat(E_low, 1), 
                         np.repeat(pos_low, observation_positions.shape),
                        np.repeat(F_low, observation_forces.shape)])
        
        high = np.hstack([np.repeat(E_high, 1), 
                         np.repeat(pos_high, observation_positions.shape),
                        np.repeat(F_high, observation_forces.shape)])
        
        
        
        return spaces.Box(low = low, high = high)
        