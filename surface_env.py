from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ase.build import fcc111
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.eos import EquationOfState as eq
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.io import read, write
from tensorforce.environments import Environment


ACTION_NAMES = ['up', 'down', 'left', 'right', 'forward', 'backward']
ACTION_DIRECTION = [
    np.array([ 0, 0, 1]), 
    np.array([ 0, 0,-1]), 
    np.array([ 1, 0, 0]), 
    np.array([-1, 0, 0]), 
    np.array([ 0, 1, 0]), 
    np.array([ 0,-1, 0])
]

class Surface():
    def __init__(self):
        # self.atom_list = []
        # with open('permutation_0.txt', 'r') as filehandle:  
        #     self.atom_list = [int(n) for n in filehandle.readlines()]
        # elements = ['Al', 'Ni', 'Cu', 'Pt']
        # mole_frac = [16, 16, 16, 16]

        # composition = [elements[0]] * mole_frac[0] + \
        #     [elements[1]] * mole_frac[1] + \
        #     [elements[2]] * mole_frac[2] + \
        #     [elements[3]] * mole_frac[3]

        # # Lattice Constants
        # Al = 4.067
        # Ni = 3.512
        # Cu = 3.626
        # Pt = 3.977

        # a = 0.25 * Al + 0.25 * Ni + 0.25 * Cu + 0.25 * Pt

        self.atom_list = []
        with open('permutation.txt', 'r') as filehandle:  
            self.atom_list = [int(n) for n in filehandle.readlines()]
        elements = ['Ag', 'Au', 'Cu']
        mole_frac = [6, 5, 5]        
        composition = [elements[0]] * mole_frac[0] + [elements[1]] * mole_frac[1] + [elements[2]] * mole_frac[2]

        # Lattice Constants
        Ag = 4.124
        Au = 4.153
        Cu = 3.626

        a = 0.33 * Ag + 0.33 * Au + 0.33 * Cu

        # slab = fcc111('Al', size=(4, 4, 4), a=a, periodic=True, vacuum=10.0)
        slab = fcc111('Al', size = (2, 2, 4), a=a, periodic=True, vacuum=10.0)
        for i in range(len(composition)):
            slab[self.atom_list[i]].symbol = composition[i]
        slab.set_calculator(EMT())

        # c =  FixAtoms(indices=[atom.index for atom in slab if atom.index < 32])  #Fix two layers
        c =  FixAtoms(indices=[atom.index for atom in slab if atom.index < 8])  #Fix two layers
        slab.set_constraint(c)
        
        self.atoms = slab
        self.free_atoms = list(set(range(len(self.atoms))) - 
            set(self.atoms.constraints[0].get_indices()))

        # print(self.free_atoms)
    
    def calculate_energy(self):
        return self.atoms.get_potential_energy()

    def max_positions(self):
        print("position shape:", self.atoms.positions.shape)
        max_x = np.max(self.atoms.positions[:,0])
        max_y = np.max(self.atoms.positions[:,1])
        max_z = np.max(self.atoms.positions[:,2])
        return max_x, max_y, max_z

    def min_positions(self):
        print("position shape:", self.atoms.positions.shape)
        min_x = np.min(self.atoms.positions[:,0])
        min_y = np.min(self.atoms.positions[:,1])
        min_z = np.min(self.atoms.positions[:,2])
        return min_x, min_y, min_z

    def current_positions(self):
        return self.atoms.positions[self.free_atoms].reshape(-1)

    def change_positions(self, idx, move):
        idx += 8
        curr_pos = self.current_positions()
        self.atoms.positions[idx] += move
        after_pos = self.current_positions()
        # print("state diff:", np.sum(after_pos - curr_pos))
        # if self.atoms.positions[idx, 0] >= 16 or self.atoms.positions[idx, 0] <= -2 or \
        #    self.atoms.positions[idx, 1] >= 11 or self.atoms.positions[idx, 1] <= -2 or \
        #    self.atoms.positions[idx, 2] >= 19 or self.atoms.positions[idx, 2] <= 8:
        #     self.atoms.positions[idx] -= move
        if self.atoms.positions[idx, 0] >= 7 or self.atoms.positions[idx, 0] <= -1 or \
           self.atoms.positions[idx, 1] >= 5 or self.atoms.positions[idx, 1] <= -1 or \
           self.atoms.positions[idx, 2] >= 18 or self.atoms.positions[idx, 2] <= 9:
            self.atoms.positions[idx] -= move

    def reset(self):
        self.__init__()

    def num_free_atoms(self):
        return len(self.free_atoms)

    def set_free_atoms(self, pos):
        curr_pos = self.current_positions()
        for idx, atom_idx in enumerate(self.free_atoms):
            # print(idx, atom_idx)
            self.atoms.positions[atom_idx] = pos[3*idx:3*idx+3]
        after_pos = self.current_positions()
        # print(np.sum(curr_pos - after_pos))
        # print(self.calculate_energy())

    def visualize(self):
        view(self.atoms)
        # viw_atoms = self.atoms.repeat((2,2,1))

    def save_fig(self, fn):
        # write(fn, self.atoms)
        # print("atom saved in {}".format(fn))
        view_atoms = self.atoms.repeat((2,2,1))
        write(fn, view_atoms)


class SurfaceEnv(Environment):
    def __init__(self, max_timesteps):
        self._lattice = Surface()
        self.curr_energy = self.get_energy()
        self.max_timesteps = max_timesteps
        self.terminal = False
        self.steps = 0
        self.state = np.empty(24)
        self.positions = np.empty((max_timesteps, 24))
        self.final_energy = []
    
    def get_state(self):
        return self._lattice.current_positions()

    def get_energy(self):
        return self._lattice.calculate_energy()

    def get_positions(self):
        return self.positions

    @property
    def states(self):
        return dict(shape=(self._lattice.num_free_atoms()*3,), type='float')

    @property
    def actions(self):
        # return dict(num_actions=32+len(ACTION_NAMES), type='int')
        return dict(num_actions=8*len(ACTION_NAMES), type='int')

    def reset(self):
        self._lattice.reset()
        self.terminal = False
        self.steps = 0
        self.positions = np.empty((self.max_timesteps, 24))
        self.state = self.get_state()
        self.positions[self.steps,:] = self.state
        self.init_energy = self.get_energy()
        return self.state

    def do_action(self, action):
        atom_idx = action // len(ACTION_DIRECTION)
        act_idx = action % len(ACTION_DIRECTION)
        # atom_select = action[:32]
        # atom_idx = np.argmax(atom_select)
        # action_select = action[32:]
        # act_idx = np.argmax(action_select)
        move = 0.2 * ACTION_DIRECTION[act_idx]
        self._lattice.change_positions(atom_idx, move)
        after_energy = self.get_energy()
        return after_energy

    def execute(self, action):
        self.steps += 1
        reward = 0

        after_energy = self.do_action(action)
        # reward = -after_energy
        # self.curr_energy = after_energy
        # reward = -self.curr_energy * 0.1
        # reward = -(after_energy - self.init_energy) * 0.001
        reward = -after_energy * 0.002
        self.curr_energy = after_energy

        # Terminal
        if self.steps >= self.max_timesteps - 1:
            print('Episode ends and final energy:', self.curr_energy)
            self.terminal = True
            self.final_energy.append(self.curr_energy)
            # reward *= 100
            # reward = -after_energy
            # reward = -self.curr_energy * 0.5
            reward += -self.curr_energy * 0.1
        
        # reward += (self.init_energy - self.curr_energy) * 0.001
        # print("state diff:", np.sum(self.state - self.get_state()))
        self.state = self.get_state()
        self.positions[self.steps,:] = self.state
        # print("state diff:", np.sum(self.positions[self.steps,:] - self.positions[self.steps-1,:]))

        return self.state, self.terminal, reward

    def visualize(self):
        self._lattice.visualize()
        # view(self._lattice.atoms)

