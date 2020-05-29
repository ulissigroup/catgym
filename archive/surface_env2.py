from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.io.trajectory import Trajectory
from ase.io import read, write

from sella import Sella
import math
import itertools


ACTION_NAMES = [
    'up', 
    'down', 
    'left', 
    'right', 
    'forward', 
    'backward', 
    'steepest_descent', 
    'steepest_ascent'
]
ACTION_DIRECTION = [
    np.array([ 0, 0, 1]), 
    np.array([ 0, 0,-1]), 
    np.array([ 1, 0, 0]), 
    np.array([-1, 0, 0]), 
    np.array([ 0, 1, 0]), 
    np.array([ 0,-1, 0]),
    np.array([ 0,0., 0]),
    np.array([ 0,0., 0]),
]

element_lattice_constants = {'Ag':4.124,'Au':4.153,'Cu':3.626}

class MultiComponentSurface():
    def __init__(self, size=(2,2,4), 
                 permute_seed=42, 
                 element_choices = {'Ag': 6, 'Au':5, 'Cu':5}):

        #generate a pseudo-random sequence of elements
        np.random.seed(permute_seed)
        num_atoms = math.prod(size)
        atom_ordering = list(itertools.chain.from_iterable([[key]*element_choices[key] for key in element_choices]))
        element_list = np.random.permutation(atom_ordering)
                             
        #Use vergard's law to estimate the lattice constant
        a = np.sum([element_lattice_constants[key]*element_choices[key]/num_atoms for key in element_choices])
        
        #Generate a base FCC slab
        slab = fcc111('Al', size=size, a=a, periodic=True, vacuum=10.0)
        slab.set_chemical_symbols(element_list)
        
        #Set the calculator
        slab.set_calculator(EMT())

        #Constrain the bottom two layers
        c =  FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < np.mean(slab.positions[:,2])])  #Fix two layers
        slab.set_constraint(c)

        #Mark the free atoms
        self.atoms = slab
        self.free_atoms = list(set(range(len(self.atoms))) - 
            set(self.atoms.constraints[0].get_indices()))
        
        #Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=self.atoms, logfile=None)
        dyn.run(0.1)

    
    def calculate_energy(self):
        return self.atoms.get_potential_energy()

    def current_positions(self):
        self.atoms.wrap()
        return self.atoms.positions[self.free_atoms].reshape(-1)

    def change_positions(self, idx, move):
        self.atoms.positions[self.free_atoms[idx]] += move

    def reset(self):
        self.__init__()

    def get_atoms(self):
        return self.atoms

    def num_free_atoms(self):
        return len(self.free_atoms)

    def set_free_atoms(self, pos):
        curr_pos = self.current_positions()
        for idx, atom_idx in enumerate(self.free_atoms):
            self.atoms.positions[atom_idx] = pos[3*idx:3*idx+3]
        after_pos = self.current_positions()

    def visualize(self):
        view(self.atoms)

    def save_fig(self, fn):
        view_atoms = self.atoms.repeat((2,2,1))
        write(fn, view_atoms)


class SurfaceEnv(Environment):
    def __init__(self, max_timesteps, surface=MultiComponentSurface()):
        self._lattice =surface
        self.curr_energy = self.get_energy()
        self.initial_energy = self.get_energy()
        self.max_timesteps = max_timesteps
        self.terminal = False
        self.steps = 0
        self.state = np.empty(self._lattice.num_free_atoms()*3)
        self.positions = np.empty((max_timesteps, self._lattice.num_free_atoms()*3))
        self.final_energy = []
        self.init_atoms = self._lattice.atoms.copy()
        self.ts_energy = []
        
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
        # return dict(num_actions=self._lattice.num_free_atoms()*len(ACTION_NAMES)+2, type='int')
        return dict(num_actions=self._lattice.num_free_atoms()*len(ACTION_NAMES), type='int')

    def reset(self):
        self._lattice.reset()
        self.terminal = False
        self.steps = 0
        self.max_energy = 0.
        self.max_atoms = self._lattice.get_atoms()
        self.positions = np.empty((self.max_timesteps, 24))
        self.atom_trajs = []
        self.atom_trajs.append(self._lattice.get_atoms())
        self.state = self.get_state()
        self.positions[self.steps,:] = self.state
        self.init_energy = self.get_energy()
        self.c = 0
        self.energies = [self.init_energy]
        return self.state

    def do_action(self, action):
        atom_idx = action // len(ACTION_DIRECTION)
        act_idx = action % len(ACTION_DIRECTION)
       
        self.prev_lattice = self._lattice
        
        if action==self.actions['num_actions']-2:
            
            fix = self._lattice.atoms.constraints[0].get_indices()

            dyn = Sella(self._lattice.atoms,  # Your Atoms object
                         constraints=dict(fix=fix),  # Your constraints
                         trajectory='saddle.traj',  # Optional trajectory
                         )
            dyn.run(1e-2, steps = 100)
            
            atoms_saddle = read("saddle.traj@-1")
            
            self._lattice.atoms.positions = atoms_saddle.positions
            
        elif action==self.actions['num_actions']-1:
            
            dyn = BFGSLineSearch(atoms=self._lattice.atoms, logfile=None, trajectory='relax.traj')
            dyn.run(1e-2, steps = 100)
            
            atoms_relax = read("relax.traj@-1")

            self._lattice.atoms.positions = atoms_relax.positions
            
        else:
            if act_idx == 6:
                force = self._lattice.atoms.get_forces()[self._lattice.free_atoms,:]
                move = -0.1*force[atom_idx]
            if act_idx == 7:
                force = self._lattice.atoms.get_forces()[self._lattice.free_atoms,:]
                move = 0.1*force[atom_idx]
            else:
                move = 0.2 * ACTION_DIRECTION[act_idx]

            self._lattice.change_positions(atom_idx, move)
#         move = 0.2 * ACTION_DIRECTION[act_idx]
#         self._lattice.change_positions(atom_idx, move)

        after_energy = self.get_energy()
        return after_energy

    def execute(self, action):
        
        self.steps += 1
        reward = 0
        # cur_dist = np.linalg.norm(min_diff(self.init_atoms, self._lattice.atoms))
               
        # Terminal
        if self.steps >= self.max_timesteps - 1:
            dyn = BFGSLineSearch(atoms=self._lattice.atoms, logfile=None)
            dyn.run(1e-2)
            self.curr_energy = self.get_energy()
            
            print('Episode ends and final energy: {} max energy: {}'.format(self.curr_energy, self.max_energy))
        
            self.terminal = True
            self.final_energy.append(self.curr_energy)
            self.ts_energy.append(self.max_energy)
            # reward *= 100
            # reward = -after_energy
            # reward = -self.curr_energy * 0.5
#             reward += -(self.curr_energy-self.initial_energy)*10
            if self.curr_energy >= self.initial_energy:# and self.max_energy > 10:
                reward -= (self.curr_energy-self.initial_energy)
            elif self.curr_energy <= self.initial_energy:# and self.max_energy < 10:
                reward += (self.initial_energy-self.curr_energy)
            # new_dist = np.linalg.norm(min_diff(self.init_atoms, self._lattice.atoms))
        else:
            after_energy = self.do_action(action)
            # reward = -after_energy
            # self.curr_energy = after_energy
            # reward = -self.curr_energy * 0.1
#             reward = -(after_energy - self.init_energy) * 0.001
#            reward = -after_energy * 0.002

#             if cur_dist > self.init_distance:
#                reward+=-(self.cur_dist-self.init_distance)
#                self.init_distance = cur_dist
            # print(np.linalg.norm(min_diff(self.init_atoms, self._lattice.atoms)))
#             distance = np.linalg.norm(min_diff(self.init_atoms, self._lattice.atoms))
            distance = np.linalg.norm(min_diff(self.prev_lattice.atoms, self._lattice.atoms))

#             if distance < 5:
#                 reward += distance*0.01
            if distance > 0 and distance < 0.3:
                reward += 0.5
                
            self.curr_energy = after_energy
    
#         new_dist = np.linalg.norm(min_diff(self.init_atoms, self._lattice.atoms))
#         print('new_dist',new_dist)

#         reward += (cur_dist-new_dist)
        # penalize the transition state
    
        if self.curr_energy < self.initial_energy:
            reward += (self.initial_energy - self.curr_energy)
            
        if self.curr_energy > 10:
            reward -= 1
        
        if self.curr_energy > self.max_energy:
            reward += -(self.curr_energy - self.max_energy)
#             reward -= 1 
            self.max_energy = self.curr_energy
            self.max_atoms = self._lattice.get_atoms()

        if self.curr_energy < self.prev_lattice.atoms.get_potential_energy():
            reward += (self.prev_lattice.atoms.get_potential_energy() - self.curr_energy)
            
#         self.c = 1
#         if self.curr_energy < 10:
#             self.c = 1
#         else:
#             reward -= self.c
#             self.c += 1

            

        
        # reward += (self.init_energy - self.curr_energy) * 0.001
        # print("state diff:", np.sum(self.state - self.get_state()))
        self.state = self.get_state()
        self.positions[self.steps,:] = self.state
        self.atom_trajs.append(self._lattice.get_atoms())
        # print("state diff:", np.sum(self.positions[self.steps,:] - self.positions[self.steps-1,:]))

        path = os.path.join('./traj_files', str(self.steps) + '.traj')
#         Trajectory(path, 'w', self._lattice.atoms)
#         self._lattice.atoms.write(path)

        self.energies.append(self.curr_energy)
        return self.state, self.terminal, reward

    def visualize(self):
        self._lattice.visualize()
        # view(self._lattice.atoms)

    def save_fig(self, save_dir):
        for i in range(self.positions.shape[0]):
            self._lattice.set_free_atoms(self.positions[i,:])
            save_fn = os.path.join(save_dir, str(i)+'.png')
            self._lattice.save_fig(save_fn)

            # view_atoms = traj.repeat((2,2,1))
            # save_fn = os.path.join(save_dir, str(i)+'.png')
            # write(save_fn, view_atoms)

        print("All atom figs have been saved to", save_dir)

    def save_traj(self, save_dir):
#         for i, traj in enumerate(self.atom_trajs):
#             Trajectory(os.path.join(save_dir, str(i) + '.traj'), 'w', traj)
        write(os.path.join(save_dir + '_%f' %self.max_energy + '.traj'), self.atom_trajs)
#         Trajectory(os.path.join(save_dir, 'transition.traj'), 'w', self.max_atoms)
#         Trajectory(os.path.join(save_dir, 'final.traj'), 'w', self._lattice.atoms)
        print("All intermediate files have been saved to", save_dir)

        
def min_diff(atoms_init, atoms_final):
    positions = (atoms_final.positions-atoms_init.positions)

    fractional = np.linalg.solve(atoms_init.get_cell(complete=True).T,
                                     positions.T).T
    if True:
        for i, periodic in enumerate(atoms_init.pbc):
            if periodic:
                # Yes, we need to do it twice.
                # See the scaled_positions.py test.
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0
                
    fractional[fractional>0.5]-=1
    return np.matmul(fractional,atoms_init.get_cell(complete=True))